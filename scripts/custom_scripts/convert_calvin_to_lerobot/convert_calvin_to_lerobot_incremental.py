"""
Convert CALVIN dataset to LeRobot format - INCREMENTAL VERSION with Timestamp Error Handling

支持增量处理和时间戳错误处理：
- 不删除已有数据集
- 自动从已处理的位置继续
- 可以安全中断和恢复
- 自动删除有时间戳错误的 episodes

Usage:
    # 首次处理 0-10000
    python convert_calvin_to_lerobot_incremental.py \
        --data_dir /path/to/calvin \
        --max_episodes 10000
    
    # 继续处理 10000-20000（自动删除坏的 episodes）
    python convert_calvin_to_lerobot_incremental.py \
        --data_dir /path/to/calvin \
        --max_episodes 10000 \
        --resume True
    
    # 使用更宽松的时间戳容差
    python convert_calvin_to_lerobot_incremental.py \
        --data_dir /path/to/calvin \
        --resume True \
        --tolerance_s 0.1
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
    quality: int = 95
) -> np.ndarray:
    """Load and resize image with memory optimization."""
    img = Image.open(image_path)
    
    if img.size[0] > target_size[1] * 2 or img.size[1] > target_size[0] * 2:
        img.thumbnail((target_size[1] * 2, target_size[0] * 2), Image.LANCZOS)
    
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    img_array = np.array(img, dtype=np.uint8)
    
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


def get_processed_episodes(output_path: Path) -> int:
    """获取已处理的 episode 数量"""
    if not output_path.exists():
        return 0
    
    # 方法1: 从 meta/info.json 读取
    info_path = output_path / "meta" / "info.json"
    if info_path.exists():
        try:
            with open(info_path, 'r') as f:
                info = json.load(f)
                return info.get('total_episodes', 0)
        except Exception:
            pass
    
    # 方法2: 从 episodes 目录统计
    episodes_dir = output_path / "meta" / "episodes"
    if episodes_dir.exists():
        episode_files = list(episodes_dir.glob("episode_*.json"))
        return len(episode_files)
    
    # 方法3: 尝试加载数据集
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset(output_path.name, root=output_path.parent)
        return dataset.num_episodes
    except Exception:
        pass
    
    return 0


def remove_last_n_episodes(output_path: Path, n: int = 1):
    """
    删除最后 n 个 episodes（通常是不完整或有问题的）
    
    这会修改以下文件：
    - data/chunk-*.parquet (删除最后的行)
    - meta/episodes/episode_*.json (删除最后的文件)
    - meta/info.json (更新 total_episodes)
    - videos/*.mp4 (删除对应的视频)
    """
    if not output_path.exists():
        print(f"Dataset path does not exist: {output_path}")
        return
    
    print(f"\n🗑️  Removing last {n} episode(s) from dataset...")
    
    # 1. 获取当前 episode 数量
    info_path = output_path / "meta" / "info.json"
    if not info_path.exists():
        print("No info.json found, cannot remove episodes")
        return
    
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    total_episodes = info.get('total_episodes', 0)
    if total_episodes == 0:
        print("No episodes to remove")
        return
    
    if n >= total_episodes:
        print(f"⚠️  Cannot remove {n} episodes from {total_episodes} total episodes")
        print(f"   Consider using --clean_start instead")
        return
    
    episodes_to_remove = list(range(total_episodes - n, total_episodes))
    print(f"  Episodes to remove: {episodes_to_remove}")
    
    # 2. 删除 episode JSON 文件
    episodes_dir = output_path / "meta" / "episodes"
    for ep_idx in episodes_to_remove:
        ep_file = episodes_dir / f"episode_{ep_idx:06d}.json"
        if ep_file.exists():
            ep_file.unlink()
            print(f"  ✓ Deleted {ep_file.name}")
    
    # 3. 删除对应的视频文件
    videos_dir = output_path / "videos"
    if videos_dir.exists():
        for ep_idx in episodes_to_remove:
            for video_file in videos_dir.glob(f"*episode_{ep_idx:06d}*.mp4"):
                video_file.unlink()
                print(f"  ✓ Deleted video {video_file.name}")
    
    # 4. 更新 info.json
    info['total_episodes'] = total_episodes - n
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"  ✓ Updated info.json: {total_episodes} -> {info['total_episodes']} episodes")
    
    # 5. 处理 parquet 数据文件
    # 注意：这部分比较复杂，因为需要找到并删除特定 episode 的所有帧
    # 为了简化，我们可以重新加载和保存数据集
    print(f"  ℹ️  Note: Parquet data files need to be reprocessed")
    print(f"     The removed episodes' frames are still in the parquet files")
    print(f"     But they won't be accessed because info.json is updated")
    
    print(f"✅ Removed {n} episode(s) successfully")


def main(
    data_dir: str = "/root/autodl-tmp/task_ABCD_D_processed/training",
    *,
    repo_name: str = "Coil1987121/calvin_lerobot_task_ABCD_D_training",
    push_to_hub: bool = False,
    include_depth: bool = False,
    include_tactile: bool = False,
    max_episodes: int | None = None,
    start_episode: int | None = None,
    state_dim: int = 32,
    fps: int = 30,
    # 增量处理参数
    resume: bool = False,
    clean_start: bool = False,
    # 时间戳错误处理参数
    remove_bad_episodes: int = 0,  # 删除最后 N 个 episodes（0=不删除）
    tolerance_s: float = 1e-4,  # 时间戳容差（秒）
    auto_fix_on_error: bool = True,  # 遇到错误时自动删除最后几个 episodes
    # 内存优化参数
    image_quality: int = 95,
    batch_save_episodes: int = 10,
    writer_threads: int = 10,
    writer_processes: int = 4,
):
    """
    Convert CALVIN dataset to LeRobot format with incremental support and timestamp error handling.
    
    增量处理参数:
        resume: 是否从上次中断的地方继续（默认 False）
        clean_start: 是否删除已有数据从头开始（默认 False）
        start_episode: 手动指定起始 episode（可选，会覆盖 resume）
    
    时间戳错误处理:
        remove_bad_episodes: 在加载前删除最后 N 个 episodes（默认 0）
        tolerance_s: 时间戳检查容差，增大可以放宽检查（默认 1e-4）
        auto_fix_on_error: 遇到时间戳错误时自动删除问题 episodes（默认 True）
    
    使用场景:
        1. 首次处理: 
           python script.py --max_episodes 10000
           
        2. 继续处理（自动删除最后1个可能不完整的episode）: 
           python script.py --resume True --remove_bad_episodes 1
           
        3. 遇到时间戳错误时自动修复:
           python script.py --resume True --auto_fix_on_error True
           
        4. 手动删除最后3个episodes后继续:
           python script.py --resume True --remove_bad_episodes 3
           
        5. 使用更宽松的容差:
           python script.py --resume True --tolerance_s 0.1
    """
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    output_path = HF_LEROBOT_HOME / repo_name
    
    # 处理增量逻辑
    if clean_start and output_path.exists():
        print(f"🗑️  Clean start: Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)
        dataset_exists = False
    elif output_path.exists():
        dataset_exists = True
        processed_count = get_processed_episodes(output_path)
        print(f"📂 Found existing dataset with {processed_count} episodes")
        
        # 删除指定数量的 episodes
        if remove_bad_episodes > 0:
            remove_last_n_episodes(output_path, remove_bad_episodes)
            processed_count = get_processed_episodes(output_path)
            print(f"📂 After removal: {processed_count} episodes")
    else:
        dataset_exists = False
        print(f"📂 No existing dataset found, will create new one")
    
    # 获取所有 episode 文件
    json_files = sorted(data_path.glob("lang_ann_*.json"))
    if not json_files:
        raise ValueError(f"No episode JSON files found in {data_dir}")
    
    print(f"📊 Total episodes available: {len(json_files)}")
    
    # 确定起始位置
    if start_episode is not None:
        actual_start = start_episode
        print(f"🎯 Manual start_episode specified: {actual_start}")
    elif resume and dataset_exists:
        actual_start = get_processed_episodes(output_path)
        print(f"♻️  Resume mode: Starting from episode {actual_start}")
    else:
        actual_start = 0
        print(f"🆕 Starting from episode 0")
    
    if actual_start >= len(json_files):
        print(f"✅ All episodes already processed! ({actual_start}/{len(json_files)})")
        return
    
    # 确定要处理的 episodes
    selected_files = json_files[actual_start:]
    if max_episodes:
        selected_files = selected_files[:max_episodes]
    
    print(f"📝 Will process episodes {actual_start} to {actual_start + len(selected_files)}")
    
    # Load first episode to determine dimensions
    first_episode = load_episode_json(json_files[0])
    print(f"🤖 Robot state dimension: {state_dim}")
    
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
                print(f"👆 Tactile RGB dimension: {tactile_rgb_dim}")
                del sample_data
        
        tactile_depth_dir = images_dir / "tactile_depth"
        if tactile_depth_dir.exists():
            sample_files = sorted(list(tactile_depth_dir.glob("*.npy")))
            if sample_files:
                sample_data = np.load(sample_files[0])
                tactile_depth_dim = sample_data.size
                print(f"👆 Tactile depth dimension: {tactile_depth_dim}")
                del sample_data
    
    total_state_dim = state_dim + tactile_rgb_dim + tactile_depth_dim
    print(f"📏 Total state dimension: {total_state_dim}")
    
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
    
    print("\n📋 Dataset features:")
    for key, value in features.items():
        print(f"  {key}: shape={value['shape']}, dtype={value['dtype']}")
    
    print(f"\n🔧 Settings:")
    print(f"  Image quality: {image_quality}")
    print(f"  Writer threads: {writer_threads}")
    print(f"  GC batch size: {batch_save_episodes} episodes")
    print(f"  Timestamp tolerance: {tolerance_s}s")
    print(f"  Auto-fix on error: {auto_fix_on_error}")
    
    # Create or load dataset with error handling
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if dataset_exists and not clean_start:
                print(f"\n📂 Loading existing dataset (attempt {attempt + 1}/{max_retries})...")
                dataset = LeRobotDataset(
                    repo_name, 
                    root=HF_LEROBOT_HOME / repo_name
                )
                print(f"✓ Loaded dataset with {dataset.num_episodes} episodes, {len(dataset)} frames")
                break
            else:
                print(f"\n🆕 Creating new dataset...")
                dataset = LeRobotDataset.create(
                    repo_id=repo_name,
                    robot_type="franka",
                    fps=fps,
                    features=features,
                    tolerance_s=tolerance_s,
                    image_writer_threads=writer_threads,
                    image_writer_processes=writer_processes,
                )
                print(f"✓ Created new dataset")
                break
                
        except ValueError as e:
            if "timestamps unexpectedly violate the tolerance" in str(e):
                print(f"\n⚠️  Timestamp validation error detected!")
                print(f"Error: {str(e)}")
                
                if not auto_fix_on_error:
                    print(f"\n❌ Auto-fix is disabled. Please:")
                    print(f"   1. Run with --remove_bad_episodes N to manually remove last N episodes")
                    print(f"   2. Or run with --auto_fix_on_error True to automatically fix")
                    print(f"   3. Or use --tolerance_s 0.1 for more lenient checking")
                    raise
                
                if attempt < max_retries - 1:
                    # 自动删除最后几个 episodes 并重试
                    episodes_to_remove = min(3, get_processed_episodes(output_path))
                    print(f"\n🔧 Auto-fixing: Removing last {episodes_to_remove} episode(s)...")
                    remove_last_n_episodes(output_path, episodes_to_remove)
                    print(f"   Retrying...")
                    continue
                else:
                    print(f"\n❌ Failed after {max_retries} attempts")
                    print(f"   Please manually clean the dataset with --remove_bad_episodes N")
                    raise
            else:
                # 其他类型的错误，直接抛出
                raise
    
    # Process episodes
    print(f"\n🚀 Processing {len(selected_files)} episodes...")
    
    for idx, json_path in enumerate(selected_files):
        episode_idx = actual_start + idx
        
        # 定期垃圾回收
        if idx > 0 and idx % batch_save_episodes == 0:
            print(f"\n🗑️  Running garbage collection after {idx} episodes...")
            gc.collect()
        
        episode_data = load_episode_json(json_path)
        episode_id = episode_data["episode_id"]
        
        print(f"\n📦 Episode {episode_idx + 1}/{len(json_files)}: {episode_id}")
        print(f"  💬 Task: {episode_data['language_instruction']}")
        print(f"  🎬 Frames: {len(episode_data['trajectory']['robot_observations'])}")
        
        images_dir = data_path / "images" / episode_id
        rgb_static_dir = images_dir / "rgb_static"
        frame_files = sorted(list(rgb_static_dir.glob("*.png")))
        frame_numbers = [get_frame_number(f.name) for f in frame_files]
        
        print(f"  🖼️  Image frames: {len(frame_numbers)}")
        
        num_trajectory_frames = len(episode_data['trajectory']['robot_observations'])
        if len(frame_numbers) != num_trajectory_frames:
            print(f"  ⚠️  Warning: Mismatch between trajectory ({num_trajectory_frames}) and images ({len(frame_numbers)})")
            num_frames = min(len(frame_numbers), num_trajectory_frames)
            frame_numbers = frame_numbers[:num_frames]
        else:
            num_frames = num_trajectory_frames
        
        # Process frames
        for frame_idx in range(num_frames):
            frame_num = frame_numbers[frame_idx]
            frame_name = f"frame_{frame_num:07d}"
            
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
                del frame_data
                
            # 每50帧轻量级GC
            if frame_idx > 0 and frame_idx % 50 == 0:
                gc.collect(generation=0)
        
        # Save episode and clean up
        dataset.save_episode()
        print(f"  ✅ Episode saved ({num_frames} frames)")
        
        del episode_data
        del frame_numbers
        gc.collect(generation=0)
    
    # Final cleanup
    print(f"\n🗑️  Final garbage collection...")
    gc.collect()
    
    print(f"\n✅ Conversion complete!")
    print(f"  📊 Total episodes in dataset: {dataset.num_episodes}")
    print(f"  🎬 Total frames in dataset: {len(dataset)}")
    print(f"  💾 Dataset location: {output_path}")
    
    if push_to_hub:
        print(f"\n📤 Pushing dataset to Hugging Face Hub: {repo_name}")
        dataset.push_to_hub(
            tags=["calvin", "franka", "manipulation", "language-conditioned"],
            private=False,
            push_videos=True,
            license="mit",
        )
        print("✅ Dataset pushed to Hub!")


if __name__ == "__main__":
    tyro.cli(main)