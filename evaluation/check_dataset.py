"""
数据集检查工具 - 用于验证LeRobot数据集格式和内容
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from PIL import Image


def check_dataset_structure(dataset_path: str):
    """检查数据集目录结构"""
    print("=" * 80)
    print("检查数据集结构")
    print("=" * 80)
    
    dataset_path = pathlib.Path(dataset_path)
    
    # 检查主目录
    if not dataset_path.exists():
        print(f"❌ 数据集路径不存在: {dataset_path}")
        return False
    
    print(f"✓ 数据集路径存在: {dataset_path}")
    
    # 检查data和meta目录
    data_path = dataset_path / "data"
    meta_path = dataset_path / "meta"
    
    if not data_path.exists():
        print(f"❌ data目录不存在: {data_path}")
        return False
    print(f"✓ data目录存在")
    
    if not meta_path.exists():
        print(f"⚠️  meta目录不存在: {meta_path}")
    else:
        print(f"✓ meta目录存在")
    
    return True


def check_metadata(dataset_path: str):
    """检查元数据文件"""
    print("\n" + "=" * 80)
    print("检查元数据")
    print("=" * 80)
    
    meta_path = pathlib.Path(dataset_path) / "meta"
    
    # 检查info.json
    info_file = meta_path / "info.json"
    if info_file.exists():
        with open(info_file, 'r') as f:
            info = json.load(f)
        
        print(f"✓ 找到info.json")
        print(f"  内容预览:")
        for key, value in list(info.items())[:10]:
            print(f"    {key}: {value}")
    else:
        print(f"⚠️  未找到info.json")
    
    # 列出meta目录下的其他文件
    if meta_path.exists():
        other_files = [f for f in meta_path.iterdir() if f.name != "info.json"]
        if other_files:
            print(f"\n  其他元数据文件:")
            for f in other_files[:5]:
                print(f"    - {f.name}")


def check_data_files(dataset_path: str):
    """检查数据文件"""
    print("\n" + "=" * 80)
    print("检查数据文件")
    print("=" * 80)
    
    data_path = pathlib.Path(dataset_path) / "data"
    
    # 查找chunk目录
    chunk_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    
    if chunk_dirs:
        print(f"✓ 找到 {len(chunk_dirs)} 个chunk目录")
        print(f"  示例: {', '.join([d.name for d in chunk_dirs[:5]])}")
        
        # 检查第一个chunk
        first_chunk = chunk_dirs[0]
        parquet_files = list(first_chunk.glob("*.parquet"))
        print(f"\n  {first_chunk.name} 中的parquet文件数: {len(parquet_files)}")
        
        return parquet_files
    else:
        # 直接在data目录下查找parquet文件
        parquet_files = list(data_path.glob("*.parquet"))
        print(f"✓ 在data目录下找到 {len(parquet_files)} 个parquet文件")
        
        return parquet_files


def check_parquet_content(parquet_files: list):
    """检查parquet文件内容"""
    print("\n" + "=" * 80)
    print("检查Parquet文件内容")
    print("=" * 80)
    
    if not parquet_files:
        print("❌ 没有找到parquet文件")
        return
    
    # 读取第一个parquet文件
    sample_file = parquet_files[0]
    print(f"\n分析文件: {sample_file.name}")
    
    try:
        df = pd.read_parquet(sample_file)
        
        print(f"✓ 成功读取parquet文件")
        print(f"  行数: {len(df)}")
        print(f"  列数: {len(df.columns)}")
        
        print(f"\n  列名:")
        for col in df.columns:
            dtype = df[col].dtype
            print(f"    - {col} ({dtype})")
        
        # 检查关键列
        print(f"\n  关键列检查:")
        
        # 图像列
        image_cols = [col for col in df.columns if "image" in col.lower()]
        if image_cols:
            print(f"    ✓ 找到图像列: {', '.join(image_cols)}")
            for img_col in image_cols[:2]:
                sample_img = df.iloc[0][img_col]
                print(f"      {img_col} 类型: {type(sample_img)}")
                if isinstance(sample_img, (bytes, str)):
                    print(f"      {img_col} 示例: {str(sample_img)[:100]}...")
        else:
            print(f"    ⚠️  未找到图像列")
        
        # 动作列
        action_cols = [col for col in df.columns if "action" in col.lower()]
        if action_cols:
            print(f"    ✓ 找到动作列: {', '.join(action_cols)}")
            for act_col in action_cols[:2]:
                sample_action = df.iloc[0][act_col]
                print(f"      {act_col} 类型: {type(sample_action)}")
                if isinstance(sample_action, (list, np.ndarray)):
                    print(f"      {act_col} 维度: {np.array(sample_action).shape}")
                    print(f"      {act_col} 示例: {sample_action}")
        else:
            print(f"    ⚠️  未找到动作列")
        
        # 状态列
        state_cols = [col for col in df.columns if "state" in col.lower()]
        if state_cols:
            print(f"    ✓ 找到状态列: {', '.join(state_cols)}")
            for state_col in state_cols[:2]:
                sample_state = df.iloc[0][state_col]
                print(f"      {state_col} 类型: {type(sample_state)}")
                if isinstance(sample_state, (list, np.ndarray)):
                    print(f"      {state_col} 维度: {np.array(sample_state).shape}")
        else:
            print(f"    ℹ️  未找到状态列（可选）")
        
        # Episode index
        if "episode_index" in df.columns:
            print(f"    ✓ 找到episode_index列")
            n_episodes = df["episode_index"].nunique()
            print(f"      文件中的episode数: {n_episodes}")
        else:
            print(f"    ℹ️  未找到episode_index列")
        
        # 显示前几行数据摘要
        print(f"\n  数据摘要 (前3行):")
        print(df.head(3).to_string())
        
        return df
        
    except Exception as e:
        print(f"❌ 读取parquet文件出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def estimate_dataset_size(dataset_path: str, sample_df: pd.DataFrame):
    """估算数据集大小"""
    print("\n" + "=" * 80)
    print("数据集统计")
    print("=" * 80)
    
    data_path = pathlib.Path(dataset_path) / "data"
    
    # 统计所有parquet文件
    all_parquet = list(data_path.rglob("*.parquet"))
    total_files = len(all_parquet)
    
    print(f"  总parquet文件数: {total_files}")
    
    if sample_df is not None and total_files > 0:
        # 估算总帧数
        avg_frames_per_file = len(sample_df)
        estimated_total_frames = total_files * avg_frames_per_file
        
        print(f"  平均每文件帧数: {avg_frames_per_file}")
        print(f"  估算总帧数: {estimated_total_frames:,}")
        
        # 如果有episode_index，统计episode数
        if "episode_index" in sample_df.columns:
            episodes_per_file = sample_df["episode_index"].nunique()
            estimated_total_episodes = total_files * episodes_per_file
            print(f"  估算总episode数: {estimated_total_episodes:,}")
    
    # 计算总大小
    total_size = sum(f.stat().st_size for f in all_parquet)
    total_size_gb = total_size / (1024 ** 3)
    print(f"  总数据大小: {total_size_gb:.2f} GB")


def generate_sample_config(dataset_path: str, df: pd.DataFrame):
    """生成示例配置"""
    print("\n" + "=" * 80)
    print("生成示例配置")
    print("=" * 80)
    
    if df is None:
        return
    
    # 推断列名
    image_cols = [col for col in df.columns if "image" in col.lower()]
    action_cols = [col for col in df.columns if "action" in col.lower()]
    state_cols = [col for col in df.columns if "state" in col.lower()]
    
    config = {
        "dataset_path": str(dataset_path),
        "image_column": image_cols[0] if image_cols else "observation.image",
        "action_column": action_cols[0] if action_cols else "action",
        "state_column": state_cols[0] if state_cols else None,
    }
    
    if action_cols:
        sample_action = np.array(df.iloc[0][action_cols[0]])
        config["action_dim"] = sample_action.shape[-1] if len(sample_action.shape) > 0 else 1
    
    print("\n建议的配置:")
    print(json.dumps(config, indent=2, ensure_ascii=False))
    
    print("\n示例运行命令:")
    print(f"""
python eval_pi0_lerobot_standalone.py \\
    --model_path /path/to/your/model.pth \\
    --dataset_path {dataset_path} \\
    --action_horizon 5 \\
    --resize_size 224 \\
    --output_path ./eval_results \\
    --max_episodes 10
""")


def main():
    parser = argparse.ArgumentParser(description="检查LeRobot数据集格式")
    parser.add_argument(
        "dataset_path",
        type=str,
        help="数据集路径"
    )
    
    args = parser.parse_args()
    
    print("\n")
    print("=" * 80)
    print("LeRobot 数据集检查工具")
    print("=" * 80)
    print(f"数据集路径: {args.dataset_path}\n")
    
    # 1. 检查目录结构
    if not check_dataset_structure(args.dataset_path):
        sys.exit(1)
    
    # 2. 检查元数据
    check_metadata(args.dataset_path)
    
    # 3. 检查数据文件
    parquet_files = check_data_files(args.dataset_path)
    
    # 4. 检查parquet内容
    sample_df = check_parquet_content(parquet_files)
    
    # 5. 统计数据集大小
    estimate_dataset_size(args.dataset_path, sample_df)
    
    # 6. 生成示例配置
    generate_sample_config(args.dataset_path, sample_df)
    
    print("\n" + "=" * 80)
    print("检查完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()