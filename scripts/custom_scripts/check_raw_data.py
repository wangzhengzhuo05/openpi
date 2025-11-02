#!/usr/bin/env python3
"""
Calvin数据集探测器 - 查看npz文件中的实际内容
"""

import numpy as np
from pathlib import Path
import argparse
from collections import Counter


def inspect_npz_file(npz_path: Path, verbose: bool = True):
    """详细检查单个npz文件"""
    try:
        data = np.load(npz_path, allow_pickle=True)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"📂 文件: {npz_path.name}")
            print(f"{'='*70}")
        
        file_info = {}
        
        for key in data.files:
            array = data[key]
            
            info = {
                'shape': array.shape if hasattr(array, 'shape') else 'N/A',
                'dtype': array.dtype if hasattr(array, 'dtype') else type(array),
                'size_mb': array.nbytes / 1024 / 1024 if hasattr(array, 'nbytes') else 0
            }
            
            file_info[key] = info
            
            if verbose:
                print(f"\n🔑 Key: '{key}'")
                print(f"   形状: {info['shape']}")
                print(f"   类型: {info['dtype']}")
                print(f"   大小: {info['size_mb']:.2f} MB")
                
                # 显示一些统计信息
                if hasattr(array, 'shape') and len(array.shape) > 0:
                    if array.dtype in [np.float32, np.float64, np.int32, np.int64]:
                        print(f"   范围: [{array.min():.4f}, {array.max():.4f}]")
                        print(f"   均值: {array.mean():.4f}")
        
        return file_info
        
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return None


def scan_dataset(dataset_path: Path, num_samples: int = 10):
    """扫描数据集，统计所有可用的键"""
    print(f"\n{'='*70}")
    print(f"🔍 扫描数据集: {dataset_path}")
    print(f"{'='*70}")
    
    npz_files = sorted(dataset_path.glob('episode_*.npz'))
    
    if not npz_files:
        print("❌ 未找到npz文件")
        return
    
    print(f"✅ 找到 {len(npz_files)} 个文件")
    print(f"📊 采样 {min(num_samples, len(npz_files))} 个文件进行分析...")
    
    # 采样文件
    sample_indices = np.linspace(0, len(npz_files)-1, num_samples, dtype=int)
    sample_files = [npz_files[i] for i in sample_indices]
    
    # 统计所有键
    all_keys = Counter()
    key_shapes = {}
    
    for npz_file in sample_files:
        try:
            data = np.load(npz_file, allow_pickle=True)
            
            for key in data.files:
                all_keys[key] += 1
                
                # 记录形状
                if key not in key_shapes:
                    array = data[key]
                    key_shapes[key] = {
                        'shape': array.shape if hasattr(array, 'shape') else 'N/A',
                        'dtype': array.dtype if hasattr(array, 'dtype') else type(array)
                    }
                    
        except Exception as e:
            print(f"⚠️  读取 {npz_file.name} 失败: {e}")
    
    # 打印统计结果
    print(f"\n{'='*70}")
    print(f"📊 数据集内容统计")
    print(f"{'='*70}")
    
    print(f"\n发现的数据键（按频率排序）:")
    print(f"{'-'*70}")
    
    for key, count in all_keys.most_common():
        percentage = (count / len(sample_files)) * 100
        shape_info = key_shapes[key]
        
        print(f"\n🔑 '{key}'")
        print(f"   出现频率: {count}/{len(sample_files)} ({percentage:.1f}%)")
        print(f"   形状: {shape_info['shape']}")
        print(f"   类型: {shape_info['dtype']}")
    
    # 分析触觉相关
    print(f"\n{'='*70}")
    print(f"🔍 触觉数据分析")
    print(f"{'='*70}")
    
    tactile_keys = [k for k in all_keys.keys() if 'tact' in k.lower() or 'touch' in k.lower()]
    
    if tactile_keys:
        print(f"✅ 找到触觉相关键:")
        for key in tactile_keys:
            print(f"   • {key}: {key_shapes[key]}")
    else:
        print(f"❌ 未找到触觉数据")
        print(f"\n可能的原因:")
        print(f"   1. 此数据集配置不包含触觉传感器")
        print(f"   2. 触觉数据在不同的文件或位置")
        print(f"   3. 需要特定的Calvin环境版本")
    
    # 检查所有可用的传感器
    print(f"\n{'='*70}")
    print(f"📷 可用传感器总结")
    print(f"{'='*70}")
    
    sensors = {
        'rgb_static': 'Static Camera RGB',
        'rgb_gripper': 'Gripper Camera RGB',
        'depth_static': 'Static Camera Depth',
        'depth_gripper': 'Gripper Camera Depth',
        'robot_obs': 'Robot State',
        'scene_obs': 'Scene Objects',
        'actions': 'Actions',
        'rel_actions': 'Relative Actions'
    }
    
    for key, name in sensors.items():
        if key in all_keys:
            count = all_keys[key]
            percentage = (count / len(sample_files)) * 100
            status = "✅" if percentage > 90 else "⚠️"
            print(f"{status} {name:25s}: {count}/{len(sample_files)} ({percentage:.1f}%)")
        else:
            print(f"❌ {name:25s}: 不可用")
    
    return all_keys, key_shapes


def compare_episodes(dataset_path: Path, ep_indices: list):
    """比较不同episode的数据内容"""
    print(f"\n{'='*70}")
    print(f"🔄 比较不同episode的内容")
    print(f"{'='*70}")
    
    for idx in ep_indices:
        npz_file = dataset_path / f'episode_{idx:07d}.npz'
        if npz_file.exists():
            print(f"\n📄 Episode {idx:07d}:")
            inspect_npz_file(npz_file, verbose=False)
            
            data = np.load(npz_file, allow_pickle=True)
            print(f"   包含的键: {', '.join(data.files)}")
        else:
            print(f"❌ Episode {idx:07d} 不存在")


def main():
    parser = argparse.ArgumentParser(
        description='Calvin数据集探测器 - 查看实际包含的数据'
    )
    
    parser.add_argument('dataset_path', type=str,
                       help='Calvin数据集路径')
    parser.add_argument('--inspect_file', type=str, default=None,
                       help='详细检查特定的npz文件')
    parser.add_argument('--scan_samples', type=int, default=20,
                       help='扫描时采样的文件数量')
    parser.add_argument('--compare_episodes', type=str, default=None,
                       help='比较多个episode，用逗号分隔，如: 0,1000,2000')
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    
    if not dataset_path.exists():
        print(f"❌ 路径不存在: {dataset_path}")
        return
    
    # 模式1: 检查特定文件
    if args.inspect_file:
        file_path = Path(args.inspect_file)
        if not file_path.exists():
            file_path = dataset_path / args.inspect_file
        
        if file_path.exists():
            inspect_npz_file(file_path, verbose=True)
        else:
            print(f"❌ 文件不存在: {file_path}")
        return
    
    # 模式2: 比较多个episode
    if args.compare_episodes:
        indices = [int(x.strip()) for x in args.compare_episodes.split(',')]
        compare_episodes(dataset_path, indices)
        return
    
    # 模式3: 扫描整个数据集（默认）
    scan_dataset(dataset_path, args.scan_samples)
    
    # 给出建议
    print(f"\n{'='*70}")
    print(f"💡 使用建议")
    print(f"{'='*70}")
    print(f"\n详细检查特定文件:")
    print(f"  python inspect_calvin_data.py {dataset_path} --inspect_file episode_0000000.npz")
    print(f"\n比较多个episode:")
    print(f"  python inspect_calvin_data.py {dataset_path} --compare_episodes 0,1000,2000")
    print(f"\n扫描更多样本:")
    print(f"  python inspect_calvin_data.py {dataset_path} --scan_samples 100")


if __name__ == '__main__':
    main()
