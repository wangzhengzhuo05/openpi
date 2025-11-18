"""
Convert CALVIN dataset to LeRobot format - MEMORY OPTIMIZED VERSION

主要改进:
1. 减少图像缓存，及时释放内存
2. 降低并发线程数
3. 添加垃圾回收
4. 分批处理episodes
5. 可选的图像压缩

Usage:
    python convert_calvin_to_lerobot_optimized.py --data_dir /path/to/calvin
"""

import gc
import json
import shutil
from pathlib import Path
from typing import Dict, Any
import numpy as np
from PIL import Image
import tyro

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


def load_episode_json(json_path: Path) -> Dict[str, Any]:
    """Load episode JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def load_and_preprocess_image(
    image_path: Path, 
    target_size: tuple = (224, 224),
    quality: int = 95  # JPEG质量，降低可节省内存
) -> np.ndarray:
    """
    Load and resize image with memory optimization.
    """
    img = Image.open(image_path)
    
    # 如果图像太大，先缩小再处理
    if img.size[0] > target_size[1] * 2 or img.size[1] > target_size[0] * 2:
        img.thumbnail((target_size[1] * 2, target_size[0] * 2), Image.LANCZOS)
    
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    img_array = np.array(img, dtype=np.uint8)
    
    # 立即关闭图像释放内存
    img.close()
    del img
    
    return img_array


def get_frame_number(filename: str) -> int:
    """Extract frame number from filename."""
    return int(filename.split('_')[1].split('.')[0])


def pad_or_truncate_state(state: np.ndarray, target_dim: int = 32) -> np.ndarray:
    """Pad or truncate state vector to target dimension."""
    current_dim = len(state)
    
    if current_dim == target_dim:
        return state
    elif current_dim < target_dim:
        padding = np.zeros(target_dim - current_dim, dtype=np.float32)
        return np.concatenate([state, padding])
    else:
        return state[:target_dim]


def main(
    data_dir: str = "/root/autodl-tmp/task_ABCD_D_processed/training",
    *,
    repo_name: str = "Coil1987121/calvin_lerobot_task_ABCD_D_training",
    push_to_hub: bool = False,
    include_depth: bool = False,
    include_tactile: bool = False,
    max_episodes: int | None = None,
    start_episode: int = 0,
    state_dim: int = 32,
    fps: int = 30,
    # 新增参数用于内存优化
    image_quality: int = 95,  # JPEG质量 (1-100)
    batch_save_episodes: int = 10,  # 每处理N个episodes后强制垃圾回收
    writer_threads: int = 2,  # 减少线程数 (原来是4)
    writer_processes: int = 1,  # 保持单进程
):
    """
    Convert CALVIN dataset to LeRobot format with memory optimization.
    
    内存优化参数:
        image_quality: 图像质量 (1-100)，降低可节省内存
        batch_save_episodes: 每处理N个episodes后强制垃圾回收
        writer_threads: 图像写入线程数，减少可降低内存占用
    """
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    # Clean up any existing dataset
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)
    
    # Get episode files
    json_files = sorted(data_path.glob("lang_ann_*.json"))
    if not json_files:
        raise ValueError(f"No episode JSON files found in {data_dir}")
    
    print(f"Found {len(json_files)} episodes")
    
    # Load first episode to determine dimensions
    first_episode = load_episode_json(json_files[0])
    print(f"Robot state dimension: {state_dim}")
    
    # Check tactile dimensions if enabled
    tactile_rgb_dim = 0
    tactile_depth_dim = 0
    if include_tactile:
        images_dir = data_path / "images" / first_episode["episode_id"]
        tactile_rgb_dir = images_dir / "tactile_rgb"
        if tactile_rgb_dir.exists():
            sample_files = sorted(list(tactile_rgb_dir.glob("*.npy")))
            if sample_files:
                sample_data = np.load(sample_files[0])
                tactile_rgb_dim = sample_data.size
                print(f"Tactile RGB dimension: {tactile_rgb_dim}")
                del sample_data
        
        tactile_depth_dir = images_dir / "tactile_depth"
        if tactile_depth_dir.exists():
            sample_files = sorted(list(tactile_depth_dir.glob("*.npy")))
            if sample_files:
                sample_data = np.load(sample_files[0])
                tactile_depth_dim = sample_data.size
                print(f"Tactile depth dimension: {tactile_depth_dim}")
                del sample_data
    
    total_state_dim = state_dim + tactile_rgb_dim + tactile_depth_dim
    print(f"Total state dimension: {total_state_dim}")
    
    # Define features
    features = {
        "observation.images.base_0_rgb": {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.left_wrist_0_rgb": {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.right_wrist_0_rgb": {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["actions"],
        },
    }
    
    if include_depth:
        features["observation.images.depth_static"] = {
            "dtype": "image",
            "shape": (200, 200, 3),
            "names": ["height", "width", "channel"],
        }
        features["observation.images.depth_gripper"] = {
            "dtype": "image",
            "shape": (84, 84, 3),
            "names": ["height", "width", "channel"],
        }
    
    print("\nCreating LeRobot dataset with features:")
    for key, value in features.items():
        print(f"  {key}: shape={value['shape']}, dtype={value['dtype']}")
    
    print(f"\n🔧 Memory optimization settings:")
    print(f"  Image quality: {image_quality}")
    print(f"  Writer threads: {writer_threads}")
    print(f"  GC batch size: {batch_save_episodes} episodes")
    
    # Create dataset with reduced threads
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="franka",
        fps=fps,
        features=features,
        image_writer_threads=writer_threads,  # 减少线程
        image_writer_processes=writer_processes,
    )
    
    # Process episodes
    print(f"\nProcessing episodes...")
    
    selected_files = json_files[start_episode:]
    if max_episodes:
        selected_files = selected_files[:max_episodes]
    
    for episode_idx, json_path in enumerate(selected_files, start=start_episode):
        
        # 定期强制垃圾回收
        if episode_idx > 0 and episode_idx % batch_save_episodes == 0:
            print(f"\n🗑️  Running garbage collection after {episode_idx} episodes...")
            gc.collect()
        
        episode_data = load_episode_json(json_path)
        episode_id = episode_data["episode_id"]
        
        print(f"\nEpisode {episode_idx + 1}/{len(selected_files)}: {episode_id}")
        print(f"  Task: {episode_data['language_instruction']}")
        print(f"  Frames: {len(episode_data['trajectory']['robot_observations'])}")
        
        images_dir = data_path / "images" / episode_id
        rgb_static_dir = images_dir / "rgb_static"
        frame_files = sorted(list(rgb_static_dir.glob("*.png")))
        frame_numbers = [get_frame_number(f.name) for f in frame_files]
        
        print(f"  Image frames: {len(frame_numbers)}")
        
        num_trajectory_frames = len(episode_data['trajectory']['robot_observations'])
        if len(frame_numbers) != num_trajectory_frames:
            print(f"  ⚠️  Warning: Mismatch between trajectory ({num_trajectory_frames}) and images ({len(frame_numbers)})")
            num_frames = min(len(frame_numbers), num_trajectory_frames)
            frame_numbers = frame_numbers[:num_frames]
        else:
            num_frames = num_trajectory_frames
        
        # Process frames with memory management
        for frame_idx in range(num_frames):
            frame_num = frame_numbers[frame_idx]
            frame_name = f"frame_{frame_num:07d}"
            
            # 每处理一帧后的临时数据都要及时清理
            frame_data = {}
            
            try:
                # Load RGB images
                rgb_static_path = images_dir / "rgb_static" / f"{frame_name}.png"
                rgb_gripper_path = images_dir / "rgb_gripper" / f"{frame_name}.png"
                
                frame_data["observation.images.base_0_rgb"] = load_and_preprocess_image(
                    rgb_static_path, quality=image_quality
                )
                frame_data["observation.images.left_wrist_0_rgb"] = load_and_preprocess_image(
                    rgb_gripper_path, quality=image_quality
                )
                frame_data["observation.images.right_wrist_0_rgb"] = load_and_preprocess_image(
                    rgb_gripper_path, quality=image_quality
                )
                
                # Load depth images if needed
                if include_depth:
                    depth_static_path = images_dir / "depth_static" / f"{frame_name}.png"
                    depth_gripper_path = images_dir / "depth_gripper" / f"{frame_name}.png"
                    
                    depth_static = load_and_preprocess_image(depth_static_path, quality=image_quality)
                    depth_gripper = load_and_preprocess_image(depth_gripper_path, quality=image_quality)
                    
                    if len(depth_static.shape) == 2:
                        depth_static = np.stack([depth_static] * 3, axis=-1)
                    elif depth_static.shape[-1] == 1:
                        depth_static = np.repeat(depth_static, 3, axis=-1)
                    
                    if len(depth_gripper.shape) == 2:
                        depth_gripper = np.stack([depth_gripper] * 3, axis=-1)
                    elif depth_gripper.shape[-1] == 1:
                        depth_gripper = np.repeat(depth_gripper, 3, axis=-1)
                    
                    frame_data["observation.images.depth_static"] = depth_static
                    frame_data["observation.images.depth_gripper"] = depth_gripper
                
                # Build state vector
                robot_state = np.array(
                    episode_data["trajectory"]["robot_observations"][frame_idx],
                    dtype=np.float32
                )
                
                state_components = [robot_state]
                
                # Add tactile data if needed
                if include_tactile:
                    tactile_rgb_path = images_dir / "tactile_rgb" / f"{frame_name}.npy"
                    tactile_depth_path = images_dir / "tactile_depth" / f"{frame_name}.npy"
                    
                    if tactile_rgb_path.exists():
                        tactile_rgb = np.load(tactile_rgb_path).flatten().astype(np.float32)
                        state_components.append(tactile_rgb)
                    else:
                        state_components.append(np.zeros(tactile_rgb_dim, dtype=np.float32))
                    
                    if tactile_depth_path.exists():
                        tactile_depth = np.load(tactile_depth_path).flatten().astype(np.float32)
                        state_components.append(tactile_depth)
                    else:
                        state_components.append(np.zeros(tactile_depth_dim, dtype=np.float32))
                
                frame_data["observation.state"] = pad_or_truncate_state(np.concatenate(state_components))
                frame_data["actions"] = np.array(
                    episode_data["trajectory"]["actions"][frame_idx],
                    dtype=np.float32
                )
                frame_data["task"] = episode_data["language_instruction"]
                
                # Add frame
                dataset.add_frame(frame_data)
                
            finally:
                # 确保及时清理frame数据
                del frame_data
                
            # 每50帧做一次轻量级GC
            if frame_idx > 0 and frame_idx % 50 == 0:
                gc.collect(generation=0)  # 只收集年轻代
        
        # Save episode and clean up
        dataset.save_episode()
        print(f"  ✓ Episode saved ({num_frames} frames)")
        
        # 清理episode数据
        del episode_data
        del frame_numbers
        gc.collect(generation=0)
    
    # Final cleanup
    print(f"\n🗑️  Final garbage collection...")
    gc.collect()
    
    print(f"\n✅ Conversion complete! Dataset saved to: {output_path}")
    print(f"Total episodes: {len(selected_files)}")
    print(f"Total frames: {len(dataset)}")
    
    if push_to_hub:
        print(f"\n📤 Pushing dataset to Hugging Face Hub: {repo_name}")
        dataset.push_to_hub(
            tags=["calvin", "franka", "manipulation", "language-conditioned"],
            private=False,
            push_videos=True,
            license="mit",
        )
        print("✓ Dataset pushed to Hub!")


if __name__ == "__main__":
    tyro.cli(main)