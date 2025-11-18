"""
Convert CALVIN dataset to LeRobot format - MEMORY OPTIMIZED VERSION WITH SAFETY & RESUME

主要改进:
1. 减少图像缓存，及时释放内存
2. 降低并发线程数
3. 添加垃圾回收
4. 分批处理episodes
5. 可选的图像压缩
6. 支持随机采样指定比例的数据
7. 🆕 安全防护：删除前确认，支持备份
8. 🆕 恢复功能：支持断点续传，分批处理

Usage:
    # 转换所有数据
    python convert_calvin_to_lerobot_safe.py --data_dir /path/to/calvin
    
    # 随机采样50%的数据
    python convert_calvin_to_lerobot_safe.py --data_dir /path/to/calvin --sample_ratio 0.5
    
    # 分批处理：先处理10000个
    python convert_calvin_to_lerobot_safe.py --data_dir /path/to/calvin --max_episodes 10000
    
    # 继续处理剩余的（自动检测并恢复）
    python convert_calvin_to_lerobot_safe.py --data_dir /path/to/calvin --resume
    
    # 不删除现有数据，直接追加
    python convert_calvin_to_lerobot_safe.py --data_dir /path/to/calvin --no_delete --resume
"""

import gc
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from PIL import Image
import tyro
from datetime import datetime

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


class ProgressTracker:
    """跟踪转换进度，支持断点续传"""
    
    def __init__(self, checkpoint_file: Path):
        self.checkpoint_file = checkpoint_file
        self.processed_episodes: List[str] = []
        self.total_frames = 0
        self.start_time = None
        self.load()
    
    def load(self):
        """加载已有的进度"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                self.processed_episodes = data.get('processed_episodes', [])
                self.total_frames = data.get('total_frames', 0)
                self.start_time = data.get('start_time')
                print(f"📂 Found existing progress: {len(self.processed_episodes)} episodes completed")
    
    def save(self):
        """保存当前进度"""
        data = {
            'processed_episodes': self.processed_episodes,
            'total_frames': self.total_frames,
            'start_time': self.start_time,
            'last_update': datetime.now().isoformat()
        }
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_episode(self, episode_id: str, num_frames: int):
        """记录已处理的episode"""
        if episode_id not in self.processed_episodes:
            self.processed_episodes.append(episode_id)
            self.total_frames += num_frames
            if self.start_time is None:
                self.start_time = datetime.now().isoformat()
            self.save()
    
    def is_processed(self, episode_id: str) -> bool:
        """检查episode是否已处理"""
        return episode_id in self.processed_episodes
    
    def clear(self):
        """清除进度记录"""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        self.processed_episodes = []
        self.total_frames = 0
        self.start_time = None


def confirm_action(prompt: str, default: bool = False) -> bool:
    """
    询问用户确认操作
    
    Args:
        prompt: 提示信息
        default: 默认选择
    
    Returns:
        用户是否确认
    """
    default_str = "[Y/n]" if default else "[y/N]"
    while True:
        response = input(f"{prompt} {default_str}: ").strip().lower()
        if response == '':
            return default
        if response in ['y', 'yes']:
            return True
        if response in ['n', 'no']:
            return False
        print("Please answer 'y' or 'n'")


def backup_dataset(output_path: Path) -> Path:
    """
    备份现有数据集
    
    Args:
        output_path: 数据集路径
    
    Returns:
        备份路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = output_path.parent / f"{output_path.name}_backup_{timestamp}"
    
    print(f"📦 Creating backup: {backup_path}")
    shutil.copytree(output_path, backup_path)
    print(f"✓ Backup created successfully")
    
    return backup_path


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


def select_episodes(
    json_files: list,
    start_episode: int = 0,
    max_episodes: int | None = None,
    sample_ratio: float | None = None,
    sample_count: int | None = None,
    random_seed: int | None = None,
) -> list:
    """Select episodes based on various criteria."""
    if sample_ratio is not None or sample_count is not None:
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        if sample_count is not None:
            num_samples = min(sample_count, len(json_files))
        elif sample_ratio is not None:
            if not 0.0 < sample_ratio <= 1.0:
                raise ValueError(f"sample_ratio must be between 0.0 and 1.0, got {sample_ratio}")
            num_samples = max(1, int(len(json_files) * sample_ratio))
        
        print(f"🎲 Random sampling {num_samples} episodes from {len(json_files)} total episodes")
        if random_seed is not None:
            print(f"   Random seed: {random_seed}")
        
        selected_files = random.sample(json_files, num_samples)
        selected_files = sorted(selected_files)
        
        return selected_files
    else:
        selected_files = json_files[start_episode:]
        if max_episodes:
            selected_files = selected_files[:max_episodes]
        return selected_files


def handle_existing_dataset(
    output_path: Path,
    no_delete: bool,
    force_delete: bool,
    create_backup: bool,
    resume: bool
) -> bool:
    """
    处理已存在的数据集
    
    Args:
        output_path: 数据集路径
        no_delete: 是否禁止删除
        force_delete: 是否强制删除（不询问）
        create_backup: 是否创建备份
        resume: 是否恢复模式
    
    Returns:
        是否应该继续（True）或退出（False）
    """
    if not output_path.exists():
        return True
    
    print(f"\n⚠️  Found existing dataset at: {output_path}")
    
    # 如果是恢复模式，不删除
    if resume:
        print("📥 Resume mode: Will continue from existing dataset")
        return True
    
    # 如果设置了不删除
    if no_delete:
        print("🔒 No-delete mode: Will append to existing dataset")
        return True
    
    # 显示数据集信息
    try:
        # 尝试统计文件数量
        total_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
        total_size_mb = total_size / (1024 * 1024)
        print(f"   Size: {total_size_mb:.2f} MB")
    except Exception as e:
        print(f"   (Unable to get size info: {e})")
    
    # 强制删除模式
    if force_delete:
        if create_backup:
            backup_dataset(output_path)
        print(f"🗑️  Force delete mode: Removing existing dataset")
        shutil.rmtree(output_path)
        return True
    
    # 询问用户
    print("\nOptions:")
    print("  1. Delete existing dataset and start fresh")
    print("  2. Create backup before deleting")
    print("  3. Append to existing dataset (resume)")
    print("  4. Cancel and exit")
    
    while True:
        choice = input("\nYour choice [1/2/3/4]: ").strip()
        
        if choice == '1':
            if confirm_action("⚠️  Are you sure you want to DELETE the existing dataset?", default=False):
                print(f"🗑️  Removing existing dataset...")
                shutil.rmtree(output_path)
                return True
            else:
                print("Operation cancelled")
                return False
        
        elif choice == '2':
            backup_dataset(output_path)
            print(f"🗑️  Removing existing dataset...")
            shutil.rmtree(output_path)
            return True
        
        elif choice == '3':
            print("📥 Continuing with existing dataset")
            return True
        
        elif choice == '4':
            print("❌ Operation cancelled by user")
            return False
        
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4")


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
    # 内存优化参数
    image_quality: int = 95,
    batch_save_episodes: int = 10,
    writer_threads: int = 2,
    writer_processes: int = 1,
    # 随机采样参数
    sample_ratio: float | None = None,
    sample_count: int | None = None,
    random_seed: int | None = 42,
    # 🆕 安全功能参数
    no_delete: bool = False,  # 禁止删除现有数据集
    force_delete: bool = False,  # 强制删除，不询问（危险！）
    create_backup: bool = False,  # 删除前创建备份
    # 🆕 恢复功能参数
    resume: bool = False,  # 从上次中断处继续
    checkpoint_dir: str | None = None,  # 检查点文件目录
):
    """
    Convert CALVIN dataset to LeRobot format with safety and resume features.
    
    内存优化参数:
        image_quality: 图像质量 (1-100)
        batch_save_episodes: 每处理N个episodes后强制垃圾回收
        writer_threads: 图像写入线程数
    
    随机采样参数:
        sample_ratio: 随机采样比例 (0.0-1.0)
        sample_count: 随机采样数量
        random_seed: 随机种子
    
    🆕 安全功能参数:
        no_delete: 禁止删除现有数据集，只追加（默认False）
        force_delete: 强制删除现有数据集，不询问（默认False，危险！）
        create_backup: 删除前创建备份（默认False）
    
    🆕 恢复功能参数:
        resume: 从上次中断处继续（默认False）
        checkpoint_dir: 检查点文件保存目录（默认使用数据集目录）
    
    使用场景:
        1. 分批处理大数据集:
           # 第一批：处理前10000个
           python script.py --max_episodes 10000
           
           # 第二批：继续处理剩余的
           python script.py --resume
        
        2. 意外中断后恢复:
           python script.py --resume
        
        3. 安全删除（带确认）:
           python script.py  # 会询问是否删除
        
        4. 强制删除（不询问，危险）:
           python script.py --force_delete
        
        5. 删除前备份:
           python script.py --create_backup
    """
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    # 设置检查点文件路径
    output_path = HF_LEROBOT_HOME / repo_name
    if checkpoint_dir is None:
        checkpoint_file = output_path.parent / f".{repo_name.replace('/', '_')}_checkpoint.json"
    else:
        checkpoint_file = Path(checkpoint_dir) / f"{repo_name.replace('/', '_')}_checkpoint.json"
    
    # 初始化进度跟踪器
    progress = ProgressTracker(checkpoint_file)
    
    # 处理已存在的数据集（安全功能）
    if not handle_existing_dataset(
        output_path=output_path,
        no_delete=no_delete,
        force_delete=force_delete,
        create_backup=create_backup,
        resume=resume
    ):
        print("\n❌ Exiting...")
        return
    
    # 如果不是恢复模式且删除了数据集，清除进度
    if not resume and not no_delete and not output_path.exists():
        progress.clear()
        print("🔄 Progress cleared for fresh start")
    
    # Get episode files
    json_files = sorted(data_path.glob("lang_ann_*.json"))
    if not json_files:
        raise ValueError(f"No episode JSON files found in {data_dir}")
    
    print(f"\n📊 Dataset info:")
    print(f"   Total episodes available: {len(json_files)}")
    if progress.processed_episodes:
        print(f"   Already processed: {len(progress.processed_episodes)} episodes")
        print(f"   Remaining: {len(json_files) - len(progress.processed_episodes)} episodes")
    
    # Load first episode to determine dimensions
    first_episode = load_episode_json(json_files[0])
    print(f"\n🤖 Robot state dimension: {state_dim}")
    
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
                print(f"   Tactile RGB dimension: {tactile_rgb_dim}")
                del sample_data
        
        tactile_depth_dir = images_dir / "tactile_depth"
        if tactile_depth_dir.exists():
            sample_files = sorted(list(tactile_depth_dir.glob("*.npy")))
            if sample_files:
                sample_data = np.load(sample_files[0])
                tactile_depth_dim = sample_data.size
                print(f"   Tactile depth dimension: {tactile_depth_dim}")
                del sample_data
    
    total_state_dim = state_dim + tactile_rgb_dim + tactile_depth_dim
    print(f"   Total state dimension: {total_state_dim}")
    
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
    
    print("\n🔧 Creating LeRobot dataset with features:")
    for key, value in features.items():
        print(f"   {key}: shape={value['shape']}, dtype={value['dtype']}")
    
    print(f"\n⚙️  Memory optimization settings:")
    print(f"   Image quality: {image_quality}")
    print(f"   Writer threads: {writer_threads}")
    print(f"   GC batch size: {batch_save_episodes} episodes")
    
    # Create or load dataset
    if output_path.exists() and (resume or no_delete):
        print(f"\n📂 Loading existing dataset from: {output_path}")
        dataset = LeRobotDataset(repo_id=repo_name)
        print(f"   Current dataset size: {len(dataset)} frames")
    else:
        print(f"\n🆕 Creating new dataset at: {output_path}")
        dataset = LeRobotDataset.create(
            repo_id=repo_name,
            robot_type="franka",
            fps=fps,
            features=features,
            image_writer_threads=writer_threads,
            image_writer_processes=writer_processes,
        )
    
    # Select episodes
    selected_files = select_episodes(
        json_files=json_files,
        start_episode=start_episode,
        max_episodes=max_episodes,
        sample_ratio=sample_ratio,
        sample_count=sample_count,
        random_seed=random_seed,
    )
    
    # Filter out already processed episodes (if resuming)
    if resume or no_delete:
        original_count = len(selected_files)
        selected_files = [f for f in selected_files if not progress.is_processed(load_episode_json(f)["episode_id"])]
        skipped = original_count - len(selected_files)
        if skipped > 0:
            print(f"\n⏭️  Skipping {skipped} already processed episodes")
    
    if not selected_files:
        print(f"\n✅ All episodes already processed! Nothing to do.")
        print(f"   Total frames in dataset: {len(dataset)}")
        return
    
    print(f"\n🚀 Processing {len(selected_files)} episodes...")
    if sample_ratio is not None or sample_count is not None:
        print(f"   Sampling strategy: ", end="")
        if sample_count is not None:
            print(f"Fixed count ({sample_count} episodes)")
        else:
            print(f"Ratio-based ({sample_ratio*100:.1f}% of total)")
    
    # Process episodes
    processed_count = 0
    for list_idx, json_path in enumerate(selected_files):
        
        # 定期强制垃圾回收
        if list_idx > 0 and list_idx % batch_save_episodes == 0:
            print(f"\n🗑️  Running garbage collection after {list_idx} episodes...")
            gc.collect()
        
        episode_data = load_episode_json(json_path)
        episode_id = episode_data["episode_id"]
        
        # 再次检查是否已处理（双重保险）
        if progress.is_processed(episode_id):
            continue
        
        # 获取原始episode索引
        original_idx = json_files.index(json_path)
        
        print(f"\n📝 Episode {list_idx + 1}/{len(selected_files)} (original #{original_idx}): {episode_id}")
        print(f"   Task: {episode_data['language_instruction']}")
        print(f"   Frames: {len(episode_data['trajectory']['robot_observations'])}")
        
        images_dir = data_path / "images" / episode_id
        rgb_static_dir = images_dir / "rgb_static"
        frame_files = sorted(list(rgb_static_dir.glob("*.png")))
        frame_numbers = [get_frame_number(f.name) for f in frame_files]
        
        print(f"   Image frames: {len(frame_numbers)}")
        
        num_trajectory_frames = len(episode_data['trajectory']['robot_observations'])
        if len(frame_numbers) != num_trajectory_frames:
            print(f"   ⚠️  Warning: Mismatch between trajectory ({num_trajectory_frames}) and images ({len(frame_numbers)})")
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
                
            # 每50帧做一次轻量级GC
            if frame_idx > 0 and frame_idx % 50 == 0:
                gc.collect(generation=0)
        
        # Save episode
        dataset.save_episode()
        
        # 记录进度
        progress.add_episode(episode_id, num_frames)
        processed_count += 1
        
        print(f"   ✓ Episode saved ({num_frames} frames) - Progress: {len(progress.processed_episodes)} total")
        
        # 清理
        del episode_data
        del frame_numbers
        gc.collect(generation=0)
    
    # Final cleanup
    print(f"\n🗑️  Final garbage collection...")
    gc.collect()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Conversion complete!")
    print(f"{'='*60}")
    print(f"Dataset location: {output_path}")
    print(f"Episodes processed this run: {processed_count}")
    print(f"Total episodes in dataset: {len(progress.processed_episodes)}")
    print(f"Total frames in dataset: {len(dataset)}")
    if len(json_files) > len(progress.processed_episodes):
        remaining = len(json_files) - len(progress.processed_episodes)
        print(f"\n📋 Remaining episodes to process: {remaining}")
        print(f"   To continue, run with --resume flag")
    else:
        print(f"\n🎉 All episodes processed!")
        print(f"   Checkpoint file can be safely deleted: {checkpoint_file}")
    
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