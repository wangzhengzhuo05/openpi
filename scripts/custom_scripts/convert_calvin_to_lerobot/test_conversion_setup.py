"""
Quick test script to verify CALVIN to LeRobot conversion setup.

This script performs a dry-run to check if all dependencies are installed
and the data can be read correctly, without actually converting the dataset.

Usage:
    python test_conversion_setup.py --data_dir /path/to/calvin/training
"""

import sys
from pathlib import Path
import tyro


def test_imports():
    """Test if all required packages are installed."""
    print("\n🔍 Testing imports...")
    
    missing_packages = []
    
    # Test LeRobot
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        print("  ✓ lerobot")
    except ImportError:
        print("  ✗ lerobot (not installed)")
        missing_packages.append("lerobot")
    
    # Test PIL
    try:
        from PIL import Image
        print("  ✓ pillow")
    except ImportError:
        print("  ✗ pillow (not installed)")
        missing_packages.append("pillow")
    
    # Test numpy
    try:
        import numpy as np
        print("  ✓ numpy")
    except ImportError:
        print("  ✗ numpy (not installed)")
        missing_packages.append("numpy")
    
    # Test tyro
    try:
        import tyro
        print("  ✓ tyro")
    except ImportError:
        print("  ✗ tyro (not installed)")
        missing_packages.append("tyro")
    
    # Test json (built-in)
    try:
        import json
        print("  ✓ json")
    except ImportError:
        print("  ✗ json")
        missing_packages.append("json")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print(f"Install with: pip install {' '.join(missing_packages)}")
        return False
    
    print("\n✅ All required packages installed!")
    return True


def test_data_structure(data_dir: str):
    """Test if data directory structure is correct."""
    print(f"\n🔍 Testing data structure in: {data_dir}")
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"  ✗ Directory does not exist: {data_dir}")
        return False
    
    print("  ✓ Directory exists")
    
    # Check for JSON files
    json_files = sorted(data_path.glob("lang_ann_*.json"))
    if not json_files:
        print("  ✗ No lang_ann_*.json files found")
        return False
    
    print(f"  ✓ Found {len(json_files)} episode JSON files")
    
    # Check images directory
    images_dir = data_path / "images"
    if not images_dir.exists():
        print("  ✗ 'images' directory not found")
        return False
    
    print("  ✓ 'images' directory exists")
    
    # Check episode directories
    episode_dirs = sorted([d for d in images_dir.iterdir() if d.is_dir() and d.name.startswith("lang_ann_")])
    if not episode_dirs:
        print("  ✗ No episode directories found in images/")
        return False
    
    print(f"  ✓ Found {len(episode_dirs)} episode directories")
    
    # Check first episode structure
    first_ep = episode_dirs[0]
    expected_sensors = [
        "rgb_static", "rgb_gripper",
        "depth_static", "depth_gripper",
        "tactile_rgb", "tactile_depth"
    ]
    
    found_sensors = [d.name for d in first_ep.iterdir() if d.is_dir()]
    missing_sensors = [s for s in expected_sensors if s not in found_sensors]
    
    if missing_sensors:
        print(f"  ⚠️  Missing sensor folders: {missing_sensors}")
    else:
        print(f"  ✓ All expected sensor folders found")
    
    print(f"  ✓ Found sensor folders: {', '.join(found_sensors)}")
    
    return True


def test_data_loading(data_dir: str):
    """Test if data can be loaded correctly."""
    print(f"\n🔍 Testing data loading...")
    
    try:
        import json
        import numpy as np
        from PIL import Image
    except ImportError as e:
        print(f"  ✗ Cannot test loading due to missing imports: {e}")
        return False
    
    data_path = Path(data_dir)
    
    # Load first JSON file
    json_files = sorted(data_path.glob("lang_ann_*.json"))
    if not json_files:
        print("  ✗ No JSON files to test")
        return False
    
    try:
        with open(json_files[0], 'r') as f:
            episode_data = json.load(f)
        print(f"  ✓ Successfully loaded JSON: {json_files[0].name}")
        
        # Check expected keys
        expected_keys = ["episode_id", "language_instruction", "task", "trajectory"]
        missing_keys = [k for k in expected_keys if k not in episode_data]
        
        if missing_keys:
            print(f"  ⚠️  Missing JSON keys: {missing_keys}")
        else:
            print(f"  ✓ JSON has all expected keys")
        
        # Print some info
        print(f"    Episode ID: {episode_data.get('episode_id', 'N/A')}")
        print(f"    Task: {episode_data.get('language_instruction', 'N/A')}")
        
        if 'trajectory' in episode_data:
            traj = episode_data['trajectory']
            if 'robot_observations' in traj:
                print(f"    Robot observations: {len(traj['robot_observations'])} frames")
                if traj['robot_observations']:
                    print(f"    State dimension: {len(traj['robot_observations'][0])}")
            if 'actions' in traj:
                print(f"    Actions: {len(traj['actions'])} frames")
                if traj['actions']:
                    print(f"    Action dimension: {len(traj['actions'][0])}")
        
    except Exception as e:
        print(f"  ✗ Failed to load JSON: {e}")
        return False
    
    # Test loading images
    episode_id = episode_data.get('episode_id', json_files[0].stem)
    images_dir = data_path / "images" / episode_id
    
    if not images_dir.exists():
        print(f"  ✗ Episode image directory not found: {images_dir}")
        return False
    
    # Test RGB static image
    rgb_static_dir = images_dir / "rgb_static"
    if rgb_static_dir.exists():
        png_files = sorted(list(rgb_static_dir.glob("*.png")))
        if png_files:
            try:
                img = Image.open(png_files[0])
                print(f"  ✓ Successfully loaded RGB image: {img.size} {img.mode}")
            except Exception as e:
                print(f"  ✗ Failed to load RGB image: {e}")
                return False
    
    # Test depth NPY file
    depth_static_dir = images_dir / "depth_static"
    if depth_static_dir.exists():
        npy_files = sorted(list(depth_static_dir.glob("*.npy")))
        if npy_files:
            try:
                arr = np.load(npy_files[0])
                print(f"  ✓ Successfully loaded depth array: shape={arr.shape} dtype={arr.dtype}")
            except Exception as e:
                print(f"  ✗ Failed to load depth array: {e}")
                return False
    
    # Test tactile NPY file
    tactile_rgb_dir = images_dir / "tactile_rgb"
    if tactile_rgb_dir.exists():
        npy_files = sorted(list(tactile_rgb_dir.glob("*.npy")))
        if npy_files:
            try:
                arr = np.load(npy_files[0])
                print(f"  ✓ Successfully loaded tactile array: shape={arr.shape} dtype={arr.dtype}")
            except Exception as e:
                print(f"  ✗ Failed to load tactile array: {e}")
                return False
    
    print("\n✅ All data loading tests passed!")
    return True


def main(data_dir: str):
    """Run all tests."""
    
    print("=" * 80)
    print("CALVIN TO LEROBOT CONVERSION - SETUP TEST")
    print("=" * 80)
    
    # Run tests
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n⚠️  Please install missing packages before continuing")
        print("Run: pip install lerobot pillow numpy tyro")
        sys.exit(1)
    
    structure_ok = test_data_structure(data_dir)
    
    if not structure_ok:
        print("\n⚠️  Data structure issues detected")
        print("Please check that your data directory is correct")
        sys.exit(1)
    
    loading_ok = test_data_loading(data_dir)
    
    if not loading_ok:
        print("\n⚠️  Data loading issues detected")
        sys.exit(1)
    
    # Final summary
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nYou're ready to convert your dataset!")
    print("\nRun the conversion with:")
    print(f"  python convert_calvin_to_lerobot.py --data_dir {data_dir}")
    print("\nOr with custom settings:")
    print(f"  python convert_calvin_to_lerobot.py \\")
    print(f"      --data_dir {data_dir} \\")
    print(f"      --repo_name your_username/calvin_lerobot \\")
    print(f"      --include_depth True \\")
    print(f"      --include_tactile True")


if __name__ == "__main__":
    tyro.cli(main)