"""
Convert CALVIN dataset to LeRobot format - FIXED VERSION

This version stores tactile data as flattened components in the state vector
to avoid compatibility issues with the datasets library.

Usage:
    python convert_calvin_to_lerobot.py --data_dir /path/to/calvin_debug_processed/training
"""

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


def load_image(image_path: Path) -> np.ndarray:
    """Load image and convert to numpy array."""
    img = Image.open(image_path)
    return np.array(img)


def get_frame_number(filename: str) -> int:
    """Extract frame number from filename like 'frame_0358656.png'."""
    return int(filename.split('_')[1].split('.')[0])


def main(
    data_dir: str,
    *,
    repo_name: str = "your_hf_username/calvin_lerobot",
    push_to_hub: bool = False,
    include_depth: bool = True,
    include_tactile: bool = True,
    fps: int = 30,
):
    """
    Convert CALVIN dataset to LeRobot format.
    
    Args:
        data_dir: Path to CALVIN dataset training directory
        repo_name: Repository name for the LeRobot dataset
        push_to_hub: Whether to push to Hugging Face Hub
        include_depth: Include depth images in the dataset
        include_tactile: Include tactile sensor data (flattened into state)
        fps: Frames per second for the dataset
    """
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)
    
    # Get first episode to determine actual dimensions
    json_files = sorted(data_path.glob("lang_ann_*.json"))
    if not json_files:
        raise ValueError(f"No episode JSON files found in {data_dir}")
    
    print(f"Found {len(json_files)} episodes")
    first_episode = load_episode_json(json_files[0])
    
    # Determine state dimension
    state_dim = len(first_episode["trajectory"]["robot_observations"][0])
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
                tactile_rgb_dim = sample_data.size  # Flatten
                print(f"Tactile RGB dimension (flattened): {tactile_rgb_dim}")
        
        tactile_depth_dir = images_dir / "tactile_depth"
        if tactile_depth_dir.exists():
            sample_files = sorted(list(tactile_depth_dir.glob("*.npy")))
            if sample_files:
                sample_data = np.load(sample_files[0])
                tactile_depth_dim = sample_data.size  # Flatten
                print(f"Tactile depth dimension (flattened): {tactile_depth_dim}")
    
    # Calculate total state dimension (robot state + flattened tactile)
    total_state_dim = state_dim + tactile_rgb_dim + tactile_depth_dim
    print(f"Total state dimension (with tactile): {total_state_dim}")
    
    # Define features for the LeRobot dataset
    features = {
        # RGB cameras
        "observation.images.rgb_static": {
            "dtype": "image",
            "shape": (200, 200, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.rgb_gripper": {
            "dtype": "image",
            "shape": (84, 84, 3),
            "names": ["height", "width", "channel"],
        },
        # Robot state + tactile (flattened)
        "observation.state": {
            "dtype": "float32",
            "shape": (total_state_dim,),
            "names": ["state"],
        },
        # Actions (from actions in JSON)
        "action": {
            "dtype": "float32",
            "shape": (7,),  # 7-DOF robot actions
            "names": ["action"],
        },
    }
    
    # Add depth images if requested
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
    
    # Create LeRobot dataset
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="franka",  # CALVIN uses Franka Panda robot
        fps=fps,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )
    
    # Process each episode
    print(f"\nProcessing episodes...")
    for episode_idx, json_path in enumerate(json_files):
        episode_data = load_episode_json(json_path)
        episode_id = episode_data["episode_id"]
        
        print(f"\nEpisode {episode_idx + 1}/{len(json_files)}: {episode_id}")
        print(f"  Task: {episode_data['language_instruction']}")
        print(f"  Frames: {len(episode_data['trajectory']['robot_observations'])}")
        
        # Get image directory for this episode
        images_dir = data_path / "images" / episode_id
        
        # Get all frame numbers from one of the image folders
        rgb_static_dir = images_dir / "rgb_static"
        frame_files = sorted(list(rgb_static_dir.glob("*.png")))
        frame_numbers = [get_frame_number(f.name) for f in frame_files]
        
        print(f"  Image frames: {len(frame_numbers)}")
        
        # Check if trajectory length matches image frames
        num_trajectory_frames = len(episode_data['trajectory']['robot_observations'])
        if len(frame_numbers) != num_trajectory_frames:
            print(f"  ⚠️  Warning: Mismatch between trajectory ({num_trajectory_frames}) and images ({len(frame_numbers)})")
            num_frames = min(len(frame_numbers), num_trajectory_frames)
            frame_numbers = frame_numbers[:num_frames]
        else:
            num_frames = num_trajectory_frames
        
        # Process each frame in the episode
        for frame_idx in range(num_frames):
            frame_num = frame_numbers[frame_idx]
            frame_name = f"frame_{frame_num:07d}"
            
            # Prepare frame data
            frame_data = {}
            
            # Load RGB images
            rgb_static_path = images_dir / "rgb_static" / f"{frame_name}.png"
            rgb_gripper_path = images_dir / "rgb_gripper" / f"{frame_name}.png"
            
            frame_data["observation.images.rgb_static"] = load_image(rgb_static_path)
            frame_data["observation.images.rgb_gripper"] = load_image(rgb_gripper_path)
            
            # Load depth images if requested
            if include_depth:
                depth_static_path = images_dir / "depth_static" / f"{frame_name}.png"
                depth_gripper_path = images_dir / "depth_gripper" / f"{frame_name}.png"
                
                depth_static = load_image(depth_static_path)
                depth_gripper = load_image(depth_gripper_path)
                
                # Convert single-channel depth to 3-channel RGB format (required by LeRobot)
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
            
            # Build state vector: robot state + flattened tactile
            robot_state = np.array(
                episode_data["trajectory"]["robot_observations"][frame_idx],
                dtype=np.float32
            )
            
            state_components = [robot_state]
            
            # Add tactile data if requested (flattened)
            if include_tactile:
                tactile_rgb_path = images_dir / "tactile_rgb" / f"{frame_name}.npy"
                tactile_depth_path = images_dir / "tactile_depth" / f"{frame_name}.npy"
                
                if tactile_rgb_path.exists():
                    tactile_rgb = np.load(tactile_rgb_path).flatten().astype(np.float32)
                    state_components.append(tactile_rgb)
                else:
                    # Pad with zeros if missing
                    state_components.append(np.zeros(tactile_rgb_dim, dtype=np.float32))
                
                if tactile_depth_path.exists():
                    tactile_depth = np.load(tactile_depth_path).flatten().astype(np.float32)
                    state_components.append(tactile_depth)
                else:
                    # Pad with zeros if missing
                    state_components.append(np.zeros(tactile_depth_dim, dtype=np.float32))
            
            # Concatenate all state components
            frame_data["observation.state"] = np.concatenate(state_components)
            
            # Add actions
            frame_data["action"] = np.array(
                episode_data["trajectory"]["actions"][frame_idx],
                dtype=np.float32
            )
            
            # Add task/language instruction
            frame_data["task"] = episode_data["language_instruction"]
            
            # Add frame to dataset
            dataset.add_frame(frame_data)
        
        # Save episode
        dataset.save_episode()
        print(f"  ✓ Episode saved ({num_frames} frames)")
    
    print(f"\n✅ Conversion complete! Dataset saved to: {output_path}")
    print(f"Total episodes: {len(json_files)}")
    print(f"Total frames: {len(dataset)}")
    
    # Optionally push to Hugging Face Hub
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