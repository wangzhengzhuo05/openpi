#!/usr/bin/env python3
"""
Deep Dataset Analyzer
Examines data structure, contents, and relationships in detail
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image


def deep_analyze(folder_path):
    """Deep analysis of the dataset"""
    
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"❌ Error: Folder '{folder_path}' does not exist!")
        return
    
    print("=" * 100)
    print(f"DEEP ANALYSIS: {folder_path}")
    print("=" * 100)
    
    # 1. Analyze JSON files in detail
    print("\n" + "=" * 100)
    print("📋 JSON FILES ANALYSIS")
    print("=" * 100)
    analyze_json_files(folder_path)
    
    # 2. Analyze episode structure
    print("\n" + "=" * 100)
    print("📂 EPISODE STRUCTURE")
    print("=" * 100)
    analyze_episodes(folder_path)
    
    # 3. Analyze sensor/camera data
    print("\n" + "=" * 100)
    print("📷 SENSOR/CAMERA DATA")
    print("=" * 100)
    analyze_sensors(folder_path)
    
    # 4. Analyze data distributions
    print("\n" + "=" * 100)
    print("📊 DATA STATISTICS")
    print("=" * 100)
    analyze_data_stats(folder_path)
    
    # 5. Check data consistency
    print("\n" + "=" * 100)
    print("✓ DATA CONSISTENCY CHECK")
    print("=" * 100)
    check_consistency(folder_path)
    
    print("\n" + "=" * 100)
    print("✅ Deep Analysis Complete!")
    print("=" * 100)


def analyze_json_files(folder_path):
    """Analyze all JSON files in detail"""
    
    json_files = sorted(folder_path.glob("*.json"))
    
    for json_file in json_files:
        print(f"\n📄 {json_file.name}")
        print("-" * 100)
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            print(f"File size: {json_file.stat().st_size / 1024:.2f} KB")
            print(f"Type: {type(data).__name__}")
            
            if isinstance(data, dict):
                print(f"\nKeys ({len(data)}):")
                for key in data.keys():
                    value = data[key]
                    if isinstance(value, dict):
                        print(f"  • {key}: dict with {len(value)} keys")
                        # Show first few keys of nested dict
                        if len(value) <= 5:
                            for k, v in value.items():
                                print(f"      - {k}: {type(v).__name__}")
                        else:
                            for k, v in list(value.items())[:3]:
                                print(f"      - {k}: {type(v).__name__}")
                            print(f"      ... and {len(value) - 3} more keys")
                    elif isinstance(value, list):
                        print(f"  • {key}: list with {len(value)} items")
                        if len(value) > 0:
                            print(f"      First item type: {type(value[0]).__name__}")
                            if isinstance(value[0], dict):
                                print(f"      First item keys: {list(value[0].keys())}")
                    else:
                        print(f"  • {key}: {value}")
                
                # Special handling for dataset_summary
                if json_file.name == "dataset_summary.json":
                    print("\n📌 Dataset Summary Details:")
                    if 'dataset_name' in data:
                        print(f"  Dataset: {data['dataset_name']}")
                    if 'total_episodes' in data:
                        print(f"  Total episodes: {data['total_episodes']}")
                    if 'sensor_config' in data:
                        print(f"  Sensors configured: {list(data['sensor_config'].keys())}")
                    if 'episodes' in data:
                        print(f"  Episode info count: {len(data['episodes'])}")
                        if len(data['episodes']) > 0:
                            first_ep = list(data['episodes'].values())[0]
                            print(f"  First episode keys: {list(first_ep.keys())}")
                
                # Special handling for lang_ann files
                if json_file.name.startswith("lang_ann_"):
                    print("\n📌 Language Annotation Details:")
                    if 'language' in data:
                        langs = data['language']
                        if isinstance(langs, dict):
                            print(f"  Language entries: {len(langs)}")
                            # Show a few examples
                            for idx, (k, v) in enumerate(list(langs.items())[:3]):
                                print(f"    [{k}]: {v}")
                        elif isinstance(langs, list):
                            print(f"  Language list length: {len(langs)}")
                            for i in range(min(3, len(langs))):
                                print(f"    [{i}]: {langs[i]}")
                    
                    # Check for common keys
                    common_keys = ['info', 'tasks', 'language', 'ann', 'start', 'end']
                    present_keys = [k for k in common_keys if k in data]
                    if present_keys:
                        print(f"  Present keys: {present_keys}")
            
            elif isinstance(data, list):
                print(f"\nList length: {len(data)}")
                if len(data) > 0:
                    print(f"First item type: {type(data[0]).__name__}")
                    if isinstance(data[0], dict):
                        print(f"First item keys: {list(data[0].keys())}")
                        print(f"\nFirst item content:")
                        for k, v in list(data[0].items())[:5]:
                            print(f"  • {k}: {v}")
        
        except Exception as e:
            print(f"  ⚠️  Error reading file: {e}")


def analyze_episodes(folder_path):
    """Analyze episode structure"""
    
    images_dir = folder_path / "images"
    
    if not images_dir.exists():
        print("No 'images' directory found")
        return
    
    # Find all episode directories
    episodes = sorted([d for d in images_dir.iterdir() if d.is_dir()])
    
    print(f"\nTotal episodes: {len(episodes)}")
    
    if len(episodes) > 0:
        print(f"\nEpisodes found:")
        for ep in episodes:
            print(f"  • {ep.name}")
        
        # Analyze first episode in detail
        print(f"\n📌 Detailed analysis of first episode: {episodes[0].name}")
        print("-" * 100)
        
        first_ep = episodes[0]
        sensors = sorted([d for d in first_ep.iterdir() if d.is_dir()])
        
        print(f"\nSensor folders ({len(sensors)}):")
        for sensor in sensors:
            files = list(sensor.glob("*"))
            file_types = defaultdict(int)
            for f in files:
                ext = f.suffix.lower()
                file_types[ext] += 1
            
            print(f"  📂 {sensor.name}:")
            print(f"      Total files: {len(files)}")
            for ext, count in sorted(file_types.items()):
                print(f"      {ext}: {count} files")
            
            # Show file naming pattern
            if files:
                sample_files = sorted([f.name for f in files])[:3]
                print(f"      Sample files: {', '.join(sample_files)}")


def analyze_sensors(folder_path):
    """Analyze sensor/camera data in detail"""
    
    images_dir = folder_path / "images"
    
    if not images_dir.exists():
        print("No 'images' directory found")
        return
    
    episodes = sorted([d for d in images_dir.iterdir() if d.is_dir()])
    
    if len(episodes) == 0:
        print("No episodes found")
        return
    
    first_ep = episodes[0]
    sensors = sorted([d for d in first_ep.iterdir() if d.is_dir()])
    
    print("\n🔍 Examining sensor data in detail:")
    
    for sensor in sensors[:3]:  # Analyze first 3 sensors in detail
        print(f"\n📷 Sensor: {sensor.name}")
        print("-" * 100)
        
        # Get all files
        png_files = sorted(list(sensor.glob("*.png")))
        npy_files = sorted(list(sensor.glob("*.npy")))
        
        print(f"PNG files: {len(png_files)}")
        print(f"NPY files: {len(npy_files)}")
        
        # Analyze PNG files
        if png_files:
            print(f"\n📸 PNG Image Analysis:")
            try:
                img = Image.open(png_files[0])
                print(f"  Resolution: {img.size}")
                print(f"  Mode: {img.mode}")
                print(f"  Format: {img.format}")
                
                # Check if all images have same size
                sizes = set()
                for png in png_files[:10]:
                    img = Image.open(png)
                    sizes.add(img.size)
                
                if len(sizes) == 1:
                    print(f"  ✓ All images have same size: {list(sizes)[0]}")
                else:
                    print(f"  ⚠️  Multiple image sizes found: {sizes}")
                
                # Show pixel value range
                img_array = np.array(Image.open(png_files[0]))
                print(f"  Pixel value range: [{img_array.min()}, {img_array.max()}]")
                print(f"  Pixel dtype: {img_array.dtype}")
                
            except Exception as e:
                print(f"  ⚠️  Error analyzing PNG: {e}")
        
        # Analyze NPY files
        if npy_files:
            print(f"\n📊 NumPy Array Analysis:")
            try:
                arr = np.load(npy_files[0])
                print(f"  Shape: {arr.shape}")
                print(f"  Dtype: {arr.dtype}")
                print(f"  Value range: [{arr.min():.4f}, {arr.max():.4f}]")
                print(f"  Mean: {arr.mean():.4f}")
                print(f"  Std: {arr.std():.4f}")
                
                # Check if all arrays have same shape
                shapes = set()
                for npy in npy_files[:10]:
                    arr = np.load(npy)
                    shapes.add(arr.shape)
                
                if len(shapes) == 1:
                    print(f"  ✓ All arrays have same shape: {list(shapes)[0]}")
                else:
                    print(f"  ⚠️  Multiple array shapes found: {shapes}")
                
            except Exception as e:
                print(f"  ⚠️  Error analyzing NPY: {e}")
        
        # Check file naming pattern
        if png_files:
            print(f"\n📝 File naming pattern:")
            print(f"  First file: {png_files[0].name}")
            print(f"  Last file: {png_files[-1].name}")
            print(f"  Total frames: {len(png_files)}")


def analyze_data_stats(folder_path):
    """Analyze data statistics"""
    
    images_dir = folder_path / "images"
    
    if not images_dir.exists():
        print("No 'images' directory found")
        return
    
    episodes = sorted([d for d in images_dir.iterdir() if d.is_dir()])
    
    print(f"\n📊 Episode Statistics:")
    print("-" * 100)
    
    # Count files per episode
    episode_stats = []
    
    for ep in episodes:
        sensors = [d for d in ep.iterdir() if d.is_dir()]
        total_files = 0
        sensor_files = {}
        
        for sensor in sensors:
            files = list(sensor.glob("*"))
            sensor_files[sensor.name] = len(files)
            total_files += len(files)
        
        episode_stats.append({
            'name': ep.name,
            'sensors': len(sensors),
            'total_files': total_files,
            'sensor_files': sensor_files
        })
    
    # Display stats
    if episode_stats:
        print(f"\nEpisode file counts:")
        for stat in episode_stats:
            print(f"  {stat['name']}: {stat['total_files']} files across {stat['sensors']} sensors")
        
        # Check if all episodes have same number of files
        file_counts = [s['total_files'] for s in episode_stats]
        if len(set(file_counts)) == 1:
            print(f"\n✓ All episodes have same number of files: {file_counts[0]}")
        else:
            print(f"\n⚠️  Episodes have different file counts:")
            print(f"  Min: {min(file_counts)}, Max: {max(file_counts)}, Avg: {sum(file_counts)/len(file_counts):.1f}")


def check_consistency(folder_path):
    """Check data consistency"""
    
    print("\n🔍 Checking data consistency...")
    
    issues = []
    
    # Check if JSON files exist
    json_files = list(folder_path.glob("*.json"))
    if not json_files:
        issues.append("⚠️  No JSON files found in root directory")
    else:
        print(f"✓ Found {len(json_files)} JSON files")
    
    # Check images directory
    images_dir = folder_path / "images"
    if not images_dir.exists():
        issues.append("⚠️  No 'images' directory found")
    else:
        print(f"✓ Images directory exists")
        
        # Check episodes
        episodes = [d for d in images_dir.iterdir() if d.is_dir()]
        if not episodes:
            issues.append("⚠️  No episode directories found in images/")
        else:
            print(f"✓ Found {len(episodes)} episode directories")
            
            # Check if episode naming is consistent
            episode_names = [e.name for e in episodes]
            if all(name.startswith("lang_ann_") for name in episode_names):
                print(f"✓ Episode naming is consistent (all start with 'lang_ann_')")
            else:
                issues.append("⚠️  Inconsistent episode naming")
            
            # Check sensor structure
            first_ep = episodes[0]
            sensors = [d.name for d in first_ep.iterdir() if d.is_dir()]
            
            all_same = True
            for ep in episodes[1:]:
                ep_sensors = [d.name for d in ep.iterdir() if d.is_dir()]
                if set(sensors) != set(ep_sensors):
                    all_same = False
                    break
            
            if all_same:
                print(f"✓ All episodes have same sensor structure ({len(sensors)} sensors)")
            else:
                issues.append("⚠️  Episodes have different sensor structures")
    
    # Print issues
    if issues:
        print(f"\n⚠️  Found {len(issues)} issues:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print(f"\n✅ No consistency issues found!")


def main():
    """Main function"""
    
    if len(sys.argv) < 2:
        print("Usage: python deep_analyze.py <folder_path>")
        print("\nExample:")
        print("  python deep_analyze.py ./my_dataset")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    deep_analyze(folder_path)


if __name__ == "__main__":
    main()