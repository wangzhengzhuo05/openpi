#!/usr/bin/env python3
"""
诊断脚本 - 检查 CALVIN 数据集和缓存配置

这个脚本帮助你：
1. 检查 lerobot 缓存目录配置
2. 查找已转换的数据集
3. 验证环境配置
"""

import os
import sys
from pathlib import Path

print("=" * 70)
print("🔍 CALVIN 数据集诊断")
print("=" * 70)

# 1. 检查 lerobot 是否安装
print("\n📦 步骤 1: 检查 lerobot 安装")
try:
    import lerobot
    print(f"   ✅ lerobot 已安装: {lerobot.__file__}")
    HAS_LEROBOT = True
except ImportError:
    print(f"   ❌ lerobot 未安装")
    print(f"   💡 请运行: pip install lerobot")
    HAS_LEROBOT = False

# 2. 检查缓存目录配置
print("\n📁 步骤 2: 检查缓存目录配置")

if HAS_LEROBOT:
    try:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
        print(f"   HF_LEROBOT_HOME: {HF_LEROBOT_HOME}")
        print(f"   存在: {HF_LEROBOT_HOME.exists()}")
        
        if HF_LEROBOT_HOME.exists():
            print(f"\n   📂 目录内容:")
            for item in HF_LEROBOT_HOME.iterdir():
                if item.is_dir():
                    print(f"      - {item.name}/")
        else:
            print(f"   ⚠️  缓存目录不存在，需要先转换数据集")
            
    except Exception as e:
        print(f"   ❌ 读取配置失败: {e}")
else:
    print(f"   ⚠️  无法检查（lerobot 未安装）")

# 3. 检查环境变量
print("\n🔧 步骤 3: 检查环境变量")
env_vars = ['HF_HOME', 'HF_DATASETS_CACHE', 'LEROBOT_HOME']
for var in env_vars:
    value = os.environ.get(var)
    if value:
        print(f"   {var}: {value}")
        print(f"   存在: {Path(value).exists()}")
    else:
        print(f"   {var}: (未设置)")

# 4. 搜索可能的数据集位置
print("\n🔍 步骤 4: 搜索 CALVIN 数据集")

search_paths = [
    Path.home() / ".cache/huggingface/lerobot",
    Path("/root/.cache/huggingface/lerobot"),
    Path("/root/autodl-tmp/huggingface/lerobot"),
    Path("/root/autodl-tmp"),
]

found_datasets = []

for search_path in search_paths:
    if not search_path.exists():
        continue
    
    print(f"\n   🔍 搜索: {search_path}")
    
    # 查找 calvin 相关目录
    try:
        for root, dirs, files in os.walk(search_path, followlinks=False):
            root_path = Path(root)
            
            # 检查是否是 LeRobot 数据集（有 data 和 meta 目录）
            if 'calvin' in root_path.name.lower():
                if (root_path / 'data').exists() and (root_path / 'meta').exists():
                    found_datasets.append(root_path)
                    print(f"      ✅ 找到: {root_path}")
            
            # 限制搜索深度
            if len(Path(root).relative_to(search_path).parts) > 3:
                dirs.clear()
                
    except Exception as e:
        print(f"      ❌ 搜索失败: {e}")

# 5. 总结
print("\n" + "=" * 70)
print("📊 诊断总结")
print("=" * 70)

if not HAS_LEROBOT:
    print("\n❌ 问题: lerobot 未安装")
    print("📝 解决方案:")
    print("   pip install lerobot")
    sys.exit(1)

if not found_datasets:
    print("\n❌ 问题: 未找到 CALVIN 数据集")
    print("\n📝 解决方案:")
    print("   1. 确保已转换数据集到 LeRobot 格式")
    print("   2. 运行转换脚本:")
    print("      python convert_calvin_to_lerobot_incremental.py \\")
    print("          --data_dir /path/to/calvin/training \\")
    print("          --repo_name Coil1987121/calvin_lerobot_task_ABCD_D_training")
    print("\n   3. 或者检查数据集是否在其他位置")
else:
    print(f"\n✅ 找到 {len(found_datasets)} 个数据集:")
    for ds in found_datasets:
        print(f"   📦 {ds}")
        
        # 尝试读取数据集信息
        try:
            info_path = ds / "meta" / "info.json"
            if info_path.exists():
                import json
                with open(info_path) as f:
                    info = json.load(f)
                print(f"      Episodes: {info.get('total_episodes', '?')}")
                print(f"      Frames: {info.get('total_frames', '?')}")
        except:
            pass
    
    print("\n💡 下一步:")
    print("   使用找到的数据集路径更新 create_subset.py")
    print("   或者设置环境变量:")
    
    if HAS_LEROBOT:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
        if found_datasets and found_datasets[0].parent != HF_LEROBOT_HOME:
            print(f"   export HF_HOME={found_datasets[0].parent.parent}")

print()