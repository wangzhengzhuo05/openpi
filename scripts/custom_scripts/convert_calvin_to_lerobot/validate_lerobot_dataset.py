"""
Simple validation script for LeRobot datasets.

Usage:
    python simple_validate.py --repo_name Coil1987121/calvin_lerobot
"""

import tyro
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import matplotlib.pyplot as plt


def validate(repo_name: str):
    """Validate a LeRobot dataset by loading and checking it."""
    
    print("=" * 80)
    print(f"VALIDATING: {repo_name}")
    print("=" * 80)
    
    # Load dataset
    print("\n📂 Loading dataset...")
    try:
        dataset = LeRobotDataset(repo_name)
        print("✓ Dataset loaded successfully!\n")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return
    
    # Print basic info
    print("📊 Dataset Information:")
    print(f"  Total frames: {len(dataset)}")
    print(f"  Number of episodes: {dataset.num_episodes}")
    print(f"  FPS: {dataset.fps}")
    print(f"  Robot type: {dataset.robot_type}")
    
    # Print features
    print(f"\n📋 Features:")
    for key in dataset.hf_dataset.features.keys():
        print(f"  • {key}")
    
    # Print episodes
    print(f"\n📂 Episodes:")
    for ep_idx in range(dataset.num_episodes):
        from_idx = dataset.episode_data_index["from"][ep_idx].item()
        to_idx = dataset.episode_data_index["to"][ep_idx].item()
        length = to_idx - from_idx
        print(f"  Episode {ep_idx}: {length} frames (index {from_idx} to {to_idx})")
    
    # Test loading a few frames
    print(f"\n🔍 Testing frame loading...")
    test_indices = [0, len(dataset) // 2, len(dataset) - 1]
    
    for idx in test_indices:
        try:
            frame = dataset[idx]
            print(f"  ✓ Frame {idx} loaded successfully")
            if idx == 0:
                print(f"    Keys: {list(frame.keys())}")
                if 'task' in frame:
                    print(f"    Task: {frame['task']}")
        except Exception as e:
            print(f"  ✗ Frame {idx} failed: {e}")
            return
    
    # Try to visualize first frame
    print(f"\n📸 Creating sample visualization...")
    try:
        frame = dataset[0]
        
        # Find image keys
        img_keys = [k for k in frame.keys() if 'image' in k.lower()]
        
        if img_keys:
            n_imgs = len(img_keys)
            fig, axes = plt.subplots(1, n_imgs, figsize=(5 * n_imgs, 5))
            if n_imgs == 1:
                axes = [axes]
            
            for ax, img_key in zip(axes, img_keys):
                img = frame[img_key].numpy()
                
                # Handle different image formats
                if img.ndim == 2:
                    ax.imshow(img, cmap='gray')
                elif img.ndim == 3:
                    if img.shape[-1] == 1:
                        ax.imshow(img[:, :, 0], cmap='gray')
                    elif img.shape[-1] == 3:
                        if img.max() > 1.0:
                            img = img.astype('uint8')
                        ax.imshow(img)
                
                title = img_key.replace('observation.images.', '')
                ax.set_title(title)
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig('dataset_sample.png', dpi=150, bbox_inches='tight')
            print("  ✓ Sample saved to: dataset_sample.png")
            plt.close()
        else:
            print("  ⚠️  No images found in dataset")
    except Exception as e:
        print(f"  ⚠️  Could not create visualization: {e}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("✅ VALIDATION COMPLETE - Dataset is ready to use!")
    print("=" * 80)
    
    # Print usage example
    print("\n💡 Usage Example:")
    print(f"""
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Load dataset
dataset = LeRobotDataset("{repo_name}")

# Get a frame
frame = dataset[0]

# Access data
rgb_static = frame['observation.images.rgb_static']
state = frame['observation.state']
action = frame['action']
task = frame['task']

print(f"Frame 0: {{task}}")
    """)


if __name__ == "__main__":
    tyro.cli(validate)