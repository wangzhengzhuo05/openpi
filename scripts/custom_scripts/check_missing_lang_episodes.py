import os
import numpy as np

# === 配置路径 ===
data_dir = "/root/autodl-tmp/dataset/calvin_debug_dataset/training"
lang_path = os.path.join(data_dir, "lang_annotations/auto_lang_ann.npy")

# === 加载语言标注 ===
lang_data = np.load(lang_path, allow_pickle=True).item()
lang_ranges = lang_data["info"]["indx"]  # e.g. [(358656, 358720), ...]

# 将所有有标注的 episode id 收集成一个集合，方便快速查找
annotated_eps = set()
for start, end in lang_ranges:
    annotated_eps.update(range(start, end))

print(f"📚 从语言标注中加载到 {len(lang_ranges)} 个区间，共覆盖 {len(annotated_eps)} 个 episode id\n")

# === 扫描 npz 文件 ===
npz_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".npz"))
total = len(npz_files)
missing = 0
covered = 0

missing_list = []

for npz in npz_files:
    try:
        ep_id = int(npz.split("_")[1].split(".")[0])
    except Exception:
        continue  # 跳过命名不规范的文件

    if ep_id not in annotated_eps:
        missing += 1
        missing_list.append(ep_id)
    else:
        covered += 1

# === 输出统计结果 ===
print("📊 检测结果：")
print(f"  ✅ 有语言标注的 episode 数量: {covered}")
print(f"  ⚠️ 无语言标注的 episode 数量: {missing}")
print(f"  总计 .npz 文件数: {total}")

if missing > 0:
    print("\n🔍 示例缺失的 episode id（前10个）:")
    print(missing_list[:10])

coverage = covered / total * 100 if total > 0 else 0
print(f"\n📈 覆盖率: {coverage:.2f}%")