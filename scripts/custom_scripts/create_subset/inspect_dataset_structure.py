#!/usr/bin/env python3
"""
数据集结构检查工具
用于详细查看 LeRobot 数据集的结构、特征和数据格式
"""

import argparse
import json
from pathlib import Path
import numpy as np

try:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
    DEFAULT_CACHE_DIR = HF_LEROBOT_HOME
    HAS_LEROBOT = True
except ImportError:
    from huggingface_hub import constants
    DEFAULT_CACHE_DIR = Path(constants.HF_HOME) / "lerobot"
    HAS_LEROBOT = False
    print("⚠️  警告: 未找到 lerobot 库")


def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subsection(title):
    """打印子标题"""
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print(f"{'─' * 80}")


def inspect_dataset_structure(repo_id: str):
    """详细检查数据集结构"""
    
    print_section(f"检查数据集: {repo_id}")
    
    # 1. 检查缓存路径
    cache_path = DEFAULT_CACHE_DIR / repo_id
    print(f"\n📁 缓存路径: {cache_path}")
    print(f"   存在: {cache_path.exists()}")
    
    if not cache_path.exists():
        print(f"\n❌ 数据集不存在于缓存中")
        return
    
    # 2. 检查文件结构
    print_subsection("文件结构")
    
    important_paths = [
        "meta/info.json",
        "meta/stats.json",
        "meta/episode_data_index.safetensors",
        "data",
    ]
    
    for rel_path in important_paths:
        full_path = cache_path / rel_path
        exists = full_path.exists()
        symbol = "✅" if exists else "❌"
        print(f"   {symbol} {rel_path}")
    
    # 3. 读取 meta/info.json
    print_subsection("数据集元信息 (meta/info.json)")
    
    info_path = cache_path / "meta" / "info.json"
    if info_path.exists():
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        print(f"\n   基本信息:")
        print(f"      数据集名称: {info.get('codebase_version', 'N/A')}")
        print(f"      Robot 类型: {info.get('robot_type', 'N/A')}")
        print(f"      FPS: {info.get('fps', 'N/A')}")
        print(f"      总 Episodes: {info.get('total_episodes', 'N/A')}")
        print(f"      总 Frames: {info.get('total_frames', 'N/A')}")
        
        print(f"\n   特征定义 (Features):")
        if 'features' in info:
            for key, feature_info in info['features'].items():
                print(f"\n      📋 {key}:")
                print(f"         类型: {feature_info.get('dtype', 'N/A')}")
                if 'shape' in feature_info:
                    print(f"         形状: {feature_info['shape']}")
                if 'names' in feature_info:
                    print(f"         维度名: {feature_info['names']}")
                if 'info' in feature_info:
                    print(f"         信息: {feature_info['info']}")
    else:
        print("   ❌ 未找到 meta/info.json")
    
    # 4. 加载数据集并检查实际数据
    if not HAS_LEROBOT:
        print("\n❌ 需要安装 lerobot 库来加载数据集")
        return
    
    print_subsection("加载数据集")
    
    try:
        import os
        old_hf_hub_offline = os.environ.get('HF_HUB_OFFLINE')
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        try:
            dataset = LeRobotDataset(repo_id, root=cache_path)
        finally:
            if old_hf_hub_offline is None:
                os.environ.pop('HF_HUB_OFFLINE', None)
            else:
                os.environ['HF_HUB_OFFLINE'] = old_hf_hub_offline
        
        print(f"   ✅ 数据集加载成功")
        print(f"   Episodes: {dataset.num_episodes}")
        print(f"   Frames: {len(dataset)}")
        print(f"   FPS: {dataset.fps}")
        
    except Exception as e:
        print(f"   ❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 检查 features 定义
    print_subsection("Features 定义 (从数据集对象)")
    
    print(f"\n   数据集的 features 属性:")
    if hasattr(dataset, 'features'):
        for key, feature in dataset.features.items():
            print(f"\n      🔑 {key}:")
            print(f"         类型: {type(feature).__name__}")
            print(f"         详情: {feature}")
    
    # 6. 检查 HuggingFace Dataset 的 features
    print_subsection("HuggingFace Dataset Features")
    
    hf_dataset = dataset.hf_dataset
    print(f"\n   HF Dataset features:")
    for key, feature in hf_dataset.features.items():
        print(f"\n      🔑 {key}:")
        print(f"         类型: {type(feature).__name__}")
        print(f"         详情: {feature}")
    
    # 7. 检查第一个样本的实际数据
    print_subsection("样本数据检查 (前3个样本)")
    
    for sample_idx in range(min(3, len(dataset))):
        print(f"\n   📦 样本 #{sample_idx}:")
        
        try:
            sample = dataset[sample_idx]
            
            # 显示所有键
            print(f"      键: {list(sample.keys())}")
            
            # 显示每个字段的详细信息
            for key, value in sample.items():
                print(f"\n      🔹 {key}:")
                print(f"         类型: {type(value)}")
                
                if isinstance(value, (np.ndarray, np.generic)):
                    print(f"         形状: {value.shape}")
                    print(f"         dtype: {value.dtype}")
                    print(f"         值范围: [{np.min(value)}, {np.max(value)}]")
                    if value.size <= 20:
                        print(f"         值: {value}")
                elif hasattr(value, 'shape'):  # torch.Tensor
                    print(f"         形状: {value.shape}")
                    print(f"         dtype: {value.dtype}")
                    print(f"         值范围: [{value.min().item()}, {value.max().item()}]")
                    if value.numel() <= 20:
                        print(f"         值: {value}")
                elif isinstance(value, dict):
                    print(f"         子键: {list(value.keys())}")
                    for sub_key, sub_value in value.items():
                        print(f"            • {sub_key}: {type(sub_value)}", end="")
                        if hasattr(sub_value, 'shape'):
                            print(f" shape={sub_value.shape} dtype={sub_value.dtype}")
                        else:
                            print()
                else:
                    print(f"         值: {value}")
                    
        except Exception as e:
            print(f"      ❌ 读取样本失败: {e}")
    
    # 8. 检查特定特征的详细信息
    print_subsection("图像特征详细检查")
    
    # 查找所有图像特征
    image_keys = []
    sample = dataset[0]
    
    def find_image_keys(d, prefix=""):
        """递归查找图像键"""
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                find_image_keys(value, full_key)
            elif hasattr(value, 'shape') and len(value.shape) == 3:
                # 可能是图像
                image_keys.append(full_key)
    
    find_image_keys(sample)
    
    if image_keys:
        print(f"\n   找到 {len(image_keys)} 个可能的图像特征:")
        for img_key in image_keys:
            print(f"\n      🖼️  {img_key}:")
            
            # 获取该特征的实际数据
            keys = img_key.split('.')
            value = sample
            for k in keys:
                value = value[k]
            
            print(f"         形状: {value.shape}")
            print(f"         dtype: {value.dtype}")
            print(f"         值范围: [{value.min()}, {value.max()}]")
            
            # 判断通道顺序
            shape = value.shape
            if shape[0] == 3 or shape[0] == 1:
                print(f"         通道顺序: 可能是 CHW (Channels, Height, Width)")
            elif shape[2] == 3 or shape[2] == 1:
                print(f"         通道顺序: 可能是 HWC (Height, Width, Channels)")
            else:
                print(f"         通道顺序: 未知")
    
    # 9. 检查 task 相关字段
    print_subsection("Task 相关字段检查")
    
    sample = dataset[0]
    
    task_related = ['task', 'task_index', 'language_instruction']
    found_task_fields = []
    
    for field in task_related:
        if field in sample:
            found_task_fields.append(field)
            print(f"\n   ✅ 找到字段: {field}")
            print(f"      类型: {type(sample[field])}")
            print(f"      值: {sample[field]}")
    
    if not found_task_fields:
        print(f"\n   ⚠️  未找到任何 task 相关字段")
        print(f"   所有字段: {list(sample.keys())}")
    
    # 10. 对比 features 定义和实际数据
    print_subsection("特征定义 vs 实际数据对比")
    
    print(f"\n   检查特征一致性:")
    
    # 从 info.json 获取期望的 features
    expected_features = set()
    if info_path.exists():
        with open(info_path, 'r') as f:
            info = json.load(f)
        if 'features' in info:
            expected_features = set(info['features'].keys())
    
    # 从实际数据获取 features
    actual_features = set(sample.keys())
    
    print(f"\n   期望的特征 (来自 meta/info.json): {len(expected_features)}")
    for f in sorted(expected_features):
        print(f"      • {f}")
    
    print(f"\n   实际的特征 (来自数据样本): {len(actual_features)}")
    for f in sorted(actual_features):
        print(f"      • {f}")
    
    missing = expected_features - actual_features
    extra = actual_features - expected_features
    
    if missing:
        print(f"\n   ⚠️  缺少的特征: {missing}")
    
    if extra:
        print(f"\n   ⚠️  额外的特征: {extra}")
    
    if not missing and not extra:
        print(f"\n   ✅ 特征完全匹配")
    
    # 11. 检查图像形状一致性
    print_subsection("图像形状一致性检查")
    
    for img_key in image_keys[:3]:  # 只检查前3个图像特征
        print(f"\n   检查 {img_key}:")
        
        # 从 features 定义获取期望形状
        if hasattr(dataset, 'features'):
            keys = img_key.split('.')
            feature_def = dataset.features
            try:
                for k in keys:
                    if hasattr(feature_def, k):
                        feature_def = getattr(feature_def, k)
                    elif isinstance(feature_def, dict):
                        feature_def = feature_def[k]
                
                if hasattr(feature_def, 'shape'):
                    expected_shape = feature_def.shape
                    print(f"      期望形状 (从 features): {expected_shape}")
                else:
                    print(f"      期望形状: 未定义")
            except:
                print(f"      期望形状: 无法获取")
        
        # 获取实际形状
        keys = img_key.split('.')
        value = sample
        for k in keys:
            value = value[k]
        
        actual_shape = value.shape
        print(f"      实际形状: {actual_shape}")
        
        # 判断是否需要转换
        if len(actual_shape) == 3:
            if actual_shape[0] in [1, 3]:
                print(f"      💡 建议: 可能需要从 CHW 转换为 HWC")
                print(f"         转换后形状: {(actual_shape[1], actual_shape[2], actual_shape[0])}")
            elif actual_shape[2] in [1, 3]:
                print(f"      ✅ 已经是 HWC 格式")
    
    print_section("检查完成")


def main():
    parser = argparse.ArgumentParser(
        description="检查 LeRobot 数据集的详细结构"
    )
    
    parser.add_argument(
        "--repo_id",
        type=str,
        default="Coil1987121/calvin_lerobot_task_ABCD_D_training",
        help="数据集的 repo_id"
    )
    
    args = parser.parse_args()
    
    inspect_dataset_structure(args.repo_id)


if __name__ == "__main__":
    main()