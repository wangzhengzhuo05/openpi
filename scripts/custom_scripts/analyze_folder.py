#!/usr/bin/env python3
"""
Simple Folder Data Analyzer
Analyzes folder structure and examines what data files contain
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

def analyze_folder(folder_path):
    """Analyze a folder and show its structure and contents"""
    
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"❌ Error: Folder '{folder_path}' does not exist!")
        return
    
    print("=" * 80)
    print(f"ANALYZING FOLDER: {folder_path}")
    print("=" * 80)
    
    # Step 1: Show folder structure
    print("\n📁 FOLDER STRUCTURE:")
    print("-" * 80)
    show_tree(folder_path)
    
    # Step 2: Count files by extension
    print("\n\n📊 FILE TYPES:")
    print("-" * 80)
    file_counts = count_files_by_type(folder_path)
    
    for ext, count in sorted(file_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ext:20s}: {count:5d} files")
    
    # Step 3: Show total statistics
    print("\n\n📈 STATISTICS:")
    print("-" * 80)
    total_files = sum(file_counts.values())
    total_size = get_folder_size(folder_path)
    print(f"  Total files: {total_files}")
    print(f"  Total size: {format_size(total_size)}")
    print(f"  File types: {len(file_counts)}")
    
    # Step 4: Examine sample files
    print("\n\n🔍 SAMPLE FILE CONTENTS:")
    print("-" * 80)
    examine_sample_files(folder_path, file_counts)
    
    print("\n" + "=" * 80)
    print("✅ Analysis Complete!")
    print("=" * 80)


def show_tree(path, prefix="", max_depth=3, current_depth=0):
    """Display folder structure as a tree"""
    
    if current_depth >= max_depth:
        return
    
    try:
        items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        
        dirs = [item for item in items if item.is_dir()]
        files = [item for item in items if item.is_file()]
        
        # Show directories first
        for i, item in enumerate(dirs):
            is_last_dir = (i == len(dirs) - 1) and len(files) == 0
            connector = "└── " if is_last_dir else "├── "
            print(f"{prefix}{connector}📂 {item.name}/")
            
            extension = "    " if is_last_dir else "│   "
            show_tree(item, prefix + extension, max_depth, current_depth + 1)
        
        # Show files (limit to first 5 per directory)
        for i, item in enumerate(files[:5]):
            is_last = i == min(4, len(files) - 1)
            connector = "└── " if is_last else "├── "
            size = format_size(item.stat().st_size)
            print(f"{prefix}{connector}📄 {item.name} ({size})")
        
        if len(files) > 5:
            print(f"{prefix}    ... and {len(files) - 5} more files")
            
    except PermissionError:
        print(f"{prefix}    [Permission Denied]")


def count_files_by_type(folder_path):
    """Count files by their extension"""
    
    file_counts = defaultdict(int)
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            ext = Path(file).suffix.lower()
            if not ext:
                ext = "[no extension]"
            file_counts[ext] += 1
    
    return dict(file_counts)


def get_folder_size(folder_path):
    """Calculate total size of folder"""
    
    total_size = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            try:
                file_path = Path(root) / file
                total_size += file_path.stat().st_size
            except:
                pass
    
    return total_size


def format_size(size_bytes):
    """Format bytes to human readable size"""
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def examine_sample_files(folder_path, file_counts):
    """Examine sample files of each type"""
    
    # Group files by extension
    files_by_ext = defaultdict(list)
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            ext = Path(file).suffix.lower()
            if not ext:
                ext = "[no extension]"
            file_path = Path(root) / file
            files_by_ext[ext].append(file_path)
    
    # Examine first file of each type
    for ext in sorted(files_by_ext.keys()):
        if files_by_ext[ext]:
            print(f"\n{ext} files (showing first file):")
            file_path = files_by_ext[ext][0]
            print(f"  Example: {file_path.name}")
            examine_file(file_path)


def examine_file(file_path):
    """Examine a single file and show its contents/structure"""
    
    ext = file_path.suffix.lower()
    
    try:
        # JSON files
        if ext == '.json':
            import json
            with open(file_path, 'r') as f:
                data = json.load(f)
                print(f"    Type: {type(data).__name__}")
                if isinstance(data, dict):
                    print(f"    Keys: {list(data.keys())[:10]}")
                elif isinstance(data, list):
                    print(f"    Length: {len(data)}")
                    if data:
                        print(f"    First item type: {type(data[0]).__name__}")
        
        # NumPy files
        elif ext in ['.npy', '.npz']:
            import numpy as np
            if ext == '.npz':
                data = np.load(file_path)
                print(f"    Keys: {list(data.keys())}")
                for key in list(data.keys())[:3]:
                    arr = data[key]
                    print(f"      {key}: shape={arr.shape}, dtype={arr.dtype}")
            else:
                arr = np.load(file_path)
                print(f"    Shape: {arr.shape}, dtype: {arr.dtype}")
        
        # HDF5 files
        elif ext in ['.hdf5', '.h5']:
            import h5py
            with h5py.File(file_path, 'r') as f:
                print(f"    Keys: {list(f.keys())}")
                for key in list(f.keys())[:5]:
                    if isinstance(f[key], h5py.Dataset):
                        print(f"      {key}: shape={f[key].shape}, dtype={f[key].dtype}")
        
        # Pickle files
        elif ext in ['.pkl', '.pickle']:
            import pickle
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                print(f"    Type: {type(data).__name__}")
                if isinstance(data, dict):
                    print(f"    Keys: {list(data.keys())[:10]}")
                elif hasattr(data, 'shape'):
                    print(f"    Shape: {data.shape}, dtype: {data.dtype}")
        
        # Image files
        elif ext in ['.png', '.jpg', '.jpeg']:
            from PIL import Image
            img = Image.open(file_path)
            print(f"    Size: {img.size}, Mode: {img.mode}")
        
        # Text files
        elif ext in ['.txt', '.csv', '.log']:
            with open(file_path, 'r') as f:
                lines = f.readlines()[:3]
                print(f"    Lines: {len(lines)} (showing first 3)")
                for i, line in enumerate(lines, 1):
                    print(f"      Line {i}: {line.strip()[:80]}")
        
        else:
            # Just show file size
            size = file_path.stat().st_size
            print(f"    Size: {format_size(size)}")
    
    except Exception as e:
        print(f"    ⚠️  Could not read: {e}")


def main():
    """Main function"""
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_folder.py <folder_path>")
        print("\nExample:")
        print("  python analyze_folder.py ./my_dataset")
        print("  python analyze_folder.py /path/to/data")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    analyze_folder(folder_path)


if __name__ == "__main__":
    main()