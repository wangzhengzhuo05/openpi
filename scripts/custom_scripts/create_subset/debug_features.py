#!/usr/bin/env python3
"""
快速调试脚本：检查 LeRobotDataset.features 的实际内容
"""

import json
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
import os

repo_id = "Coil1987121/calvin_lerobot_task_ABCD_D_training"
cache_path = HF_LEROBOT_HOME / repo_id

print("=" * 80)
print("检查 LeRobotDataset.features 的实际内容")
print("=" * 80)

# 加载数据集
print(f"\n📥 加载数据集: {repo_id}")
os.environ['HF_HUB_OFFLINE'] = '1'

try:
    dataset = LeRobotDataset(repo_id, root=cache_path)
    print(f"✅ 加载成功\n")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    exit(1)

# 1. 检查 meta/info.json 中的 features
print("=" * 80)
print("1. meta/info.json 中定义的 features")
print("=" * 80)

meta_info_path = cache_path / "meta" / "info.json"
with open(meta_info_path, 'r') as f:
    meta_info = json.load(f)

if 'features' in meta_info:
    for key, value in meta_info['features'].items():
        print(f"  • {key}")
        print(f"    - dtype: {value.get('dtype')}")
        print(f"    - shape: {value.get('shape')}")
else:
    print("  (未找到 features 定义)")

# 2. 检查 dataset.features
print("\n" + "=" * 80)
print("2. dataset.features (Python 对象)")
print("=" * 80)

if hasattr(dataset, 'features'):
    for key, value in dataset.features.items():
        print(f"  • {key}")
        print(f"    - 类型: {type(value)}")
        print(f"    - 值: {value}")
else:
    print("  (未找到 features 属性)")

# 3. 检查实际数据样本
print("\n" + "=" * 80)
print("3. 实际数据样本中的字段")
print("=" * 80)

sample = dataset[0]
print(f"总字段数: {len(sample.keys())}\n")

for key in sorted(sample.keys()):
    value = sample[key]
    print(f"  • {key}")
    print(f"    - 类型: {type(value).__name__}")
    if hasattr(value, 'shape'):
        print(f"    - 形状: {value.shape}")
    if isinstance(value, str):
        print(f"    - 值: {value[:50]}...")

# 4. 对比分析
print("\n" + "=" * 80)
print("4. 对比分析")
print("=" * 80)

meta_features = set(meta_info.get('features', {}).keys())
dataset_features = set(dataset.features.keys()) if hasattr(dataset, 'features') else set()
sample_features = set(sample.keys())

print(f"\nmeta/info.json 特征数: {len(meta_features)}")
print(f"dataset.features 特征数: {len(dataset_features)}")
print(f"样本实际字段数: {len(sample_features)}")

# 检查关键字段
print("\n关键字段检查:")
for field in ['task', 'task_index']:
    print(f"\n  {field}:")
    print(f"    - 在 meta/info.json 中: {'✅' if field in meta_features else '❌'}")
    print(f"    - 在 dataset.features 中: {'✅' if field in dataset_features else '❌'}")
    print(f"    - 在样本数据中: {'✅' if field in sample_features else '❌'}")

# 5. 找出差异
print("\n" + "=" * 80)
print("5. 特征差异")
print("=" * 80)

if dataset_features:
    only_in_meta = meta_features - dataset_features
    only_in_dataset = dataset_features - meta_features
    only_in_sample = sample_features - dataset_features - {'index', 'episode_index', 'frame_index', 'timestamp'}
    
    if only_in_meta:
        print(f"\n❗ 只在 meta/info.json 中: {only_in_meta}")
    if only_in_dataset:
        print(f"\n❗ 只在 dataset.features 中: {only_in_dataset}")
    if only_in_sample:
        print(f"\n❗ 只在样本数据中（非元数据）: {only_in_sample}")
    
    if not (only_in_meta or only_in_dataset or only_in_sample):
        print("\n✅ 所有特征定义一致")

print("\n" + "=" * 80)
print("检查完成")
print("=" * 80)