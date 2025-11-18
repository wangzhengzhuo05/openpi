# CALVIN数据集转换工具包 🚀

完整的CALVIN数据集转换解决方案，包含内存优化、随机采样、安全防护和断点续传功能。

---

## 📦 文件清单

### 脚本文件

1. **convert_calvin_to_lerobot_safe.py** ⭐ **推荐**
   - 完整功能版本
   - 包含所有优化和安全特性
   - 支持断点续传和分批处理

2. **convert_calvin_to_lerobot_optimized_with_sampling.py**
   - 带随机采样的优化版
   - 适合需要采样但不需要恢复功能的场景

3. **你的原始文件**（未包含在输出中）
   - 基础优化版本

### 文档文件

1. **README_SAFETY_AND_RESUME.md** 📖
   - 安全功能和恢复功能详细使用指南
   - 包含所有使用场景和示例
   - **如果你要使用安全版，先看这个**

2. **README_SAMPLING.md**
   - 随机采样功能说明
   - 适用于采样版和安全版

3. **VERSION_COMPARISON.md**
   - 三个版本的对比
   - 帮助你选择合适的版本
   - **不确定用哪个？先看这个**

4. **README.md** (本文件)
   - 总体概览和快速开始

---

## 🎯 快速开始

### 情况1: 我是第一次使用（数据集<5000）

```bash
# 使用安全版，最保险
python convert_calvin_to_lerobot_safe.py \
    --data_dir /path/to/calvin \
    --repo_name "your_name/calvin_dataset"
```

### 情况2: 我有大数据集（>10000 episodes）

```bash
# 第一批：处理10000个
python convert_calvin_to_lerobot_safe.py \
    --data_dir /path/to/calvin \
    --max_episodes 10000

# 第二批：继续处理
python convert_calvin_to_lerobot_safe.py \
    --data_dir /path/to/calvin \
    --resume
```

### 情况3: 我需要采样数据测试

```bash
# 采样10%快速测试
python convert_calvin_to_lerobot_safe.py \
    --data_dir /path/to/calvin \
    --sample_ratio 0.1 \
    --repo_name "test_dataset"
```

### 情况4: 程序中断了

```bash
# 直接恢复
python convert_calvin_to_lerobot_safe.py \
    --data_dir /path/to/calvin \
    --resume
```

---

## 🆕 主要功能亮点

### 🛡️ 安全防护
- ✅ 删除前自动确认
- ✅ 可选数据备份
- ✅ 禁止删除模式
- ✅ 防止误操作

### 🔄 断点续传
- ✅ 自动保存进度
- ✅ 中断后继续处理
- ✅ 智能跳过已处理数据
- ✅ 支持分批处理大数据集

### 🎲 随机采样
- ✅ 按比例采样（如50%）
- ✅ 按数量采样（如1000个）
- ✅ 可重复采样（固定种子）

### 💾 内存优化
- ✅ 图像压缩
- ✅ 及时释放内存
- ✅ 分批垃圾回收
- ✅ 降低并发线程

---

## 📚 详细文档导航

### 按使用场景

| 场景 | 推荐文档 |
|------|----------|
| 第一次使用 | [VERSION_COMPARISON.md](VERSION_COMPARISON.md) |
| 大数据集分批处理 | [README_SAFETY_AND_RESUME.md](README_SAFETY_AND_RESUME.md) |
| 随机采样 | [README_SAMPLING.md](README_SAMPLING.md) |
| 断点续传 | [README_SAFETY_AND_RESUME.md](README_SAFETY_AND_RESUME.md) |
| 版本选择 | [VERSION_COMPARISON.md](VERSION_COMPARISON.md) |

### 按功能

| 功能 | 相关文档 | 章节 |
|------|----------|------|
| 安全删除 | README_SAFETY_AND_RESUME.md | 安全功能参数 |
| 数据备份 | README_SAFETY_AND_RESUME.md | --create-backup |
| 恢复处理 | README_SAFETY_AND_RESUME.md | 场景1 |
| 分批处理 | README_SAFETY_AND_RESUME.md | 场景1 |
| 随机采样 | README_SAMPLING.md | 新增参数 |
| 内存优化 | 所有文档 | 内存优化参数 |

---

## 🔥 推荐使用方案

### 方案A: 安全至上（推荐新手）⭐

```bash
# 1. 先测试1%
python convert_calvin_to_lerobot_safe.py \
    --data_dir /path/to/calvin \
    --sample_ratio 0.01 \
    --repo_name "test"

# 2. 确认无误后处理全部
python convert_calvin_to_lerobot_safe.py \
    --data_dir /path/to/calvin \
    --repo_name "production"
```

### 方案B: 大数据集处理（>10000）

```bash
# 分批处理，每批5000
python convert_calvin_to_lerobot_safe.py \
    --data_dir /path/to/calvin \
    --max_episodes 5000 \
    --batch_save_episodes 10

# 后续批次自动恢复
while true; do
    python convert_calvin_to_lerobot_safe.py \
        --data_dir /path/to/calvin \
        --resume || break
    
    # 检查是否完成
    if grep -q "All episodes processed" conversion.log; then
        break
    fi
done
```

### 方案C: 创建数据集变体

```bash
# 训练集（80%）
python convert_calvin_to_lerobot_safe.py \
    --data_dir /path/to/calvin \
    --sample_ratio 0.8 \
    --random_seed 42 \
    --repo_name "calvin_train"

# 验证集（20%）
python convert_calvin_to_lerobot_safe.py \
    --data_dir /path/to/calvin \
    --sample_ratio 0.2 \
    --random_seed 123 \
    --repo_name "calvin_val"
```

---

## ⚙️ 常用参数速查

### 必需参数
```bash
--data_dir /path/to/calvin          # CALVIN数据目录
--repo_name "your_name/dataset"     # HuggingFace仓库名
```

### 安全参数
```bash
--resume                            # 恢复模式
--no_delete                         # 不删除，只追加
--create_backup                     # 删除前备份
--force_delete                      # 强制删除（危险）
```

### 采样参数
```bash
--sample_ratio 0.5                  # 采样50%
--sample_count 1000                 # 采样1000个
--random_seed 42                    # 固定随机种子
```

### 优化参数
```bash
--image_quality 85                  # 图像质量（默认95）
--writer_threads 2                  # 写入线程（默认2）
--batch_save_episodes 10            # GC批次大小
--max_episodes 10000                # 最大处理数量
```

---

## 💡 使用技巧

### 技巧1: 查看进度

```bash
# 方法1: 运行时会自动显示
python convert_calvin_to_lerobot_safe.py \
    --data_dir /path/to/calvin \
    --resume

# 方法2: 查看检查点文件
cat ~/.cache/huggingface/lerobot/.your_repo_checkpoint.json
```

### 技巧2: 低内存环境

```bash
# 降低图像质量和线程数
python convert_calvin_to_lerobot_safe.py \
    --data_dir /path/to/calvin \
    --image_quality 75 \
    --writer_threads 1 \
    --batch_save_episodes 5
```

### 技巧3: 后台运行

```bash
# 使用nohup
nohup python convert_calvin_to_lerobot_safe.py \
    --data_dir /path/to/calvin \
    --resume \
    > conversion.log 2>&1 &

# 查看日志
tail -f conversion.log
```

### 技巧4: 自动重试

```bash
#!/bin/bash
# retry_convert.sh

max_retries=3
count=0

while [ $count -lt $max_retries ]; do
    python convert_calvin_to_lerobot_safe.py \
        --data_dir /path/to/calvin \
        --resume
    
    if [ $? -eq 0 ]; then
        echo "Success!"
        break
    else
        count=$((count + 1))
        echo "Retry $count/$max_retries"
        sleep 60
    fi
done
```

---

## 🐛 常见问题

### Q: 我应该用哪个版本？
**A**: 如果不确定，用安全版（convert_calvin_to_lerobot_safe.py）。它包含所有功能且最安全。

### Q: 如何分批处理20000个episodes？
**A**: 
```bash
# 第一批10000
python convert_calvin_to_lerobot_safe.py --data_dir /path --max_episodes 10000

# 第二批继续
python convert_calvin_to_lerobot_safe.py --data_dir /path --resume
```

### Q: 程序崩溃了怎么办？
**A**: 
```bash
# 直接运行恢复命令
python convert_calvin_to_lerobot_safe.py --data_dir /path --resume
```

### Q: 如何查看还剩多少要处理？
**A**: 运行时会自动显示：
```
Total episodes available: 20000
Already processed: 12345 episodes
Remaining: 7655 episodes
```

### Q: 检查点文件在哪里？
**A**: 默认在 `~/.cache/huggingface/lerobot/.{repo_name}_checkpoint.json`

### Q: 可以删除检查点文件吗？
**A**: 
- 处理中：不要删除
- 完成后：可以安全删除（脚本会提示）

---

## 📊 性能参考

### 处理速度（参考）
- 小型图像（200x200）: ~50-100 episodes/分钟
- 中型图像（400x400）: ~30-50 episodes/分钟
- 包含深度图: ~20-40 episodes/分钟

### 内存占用（参考）
- 最小配置: ~2GB
- 推荐配置: ~4GB
- 包含深度图和触觉: ~6GB

### 磁盘空间（参考）
- 每个episode: ~5-20MB（取决于帧数和图像大小）
- 20000 episodes: ~100-400GB
- 备份需要相同空间

---

## 🎓 学习路径

### 新手路径
1. 阅读 [VERSION_COMPARISON.md](VERSION_COMPARISON.md) - 了解版本差异
2. 阅读 [README_SAFETY_AND_RESUME.md](README_SAFETY_AND_RESUME.md) - 学习主要功能
3. 用1%数据测试
4. 正式处理全部数据

### 进阶路径
1. 学习随机采样 - [README_SAMPLING.md](README_SAMPLING.md)
2. 学习分批处理 - [README_SAFETY_AND_RESUME.md](README_SAFETY_AND_RESUME.md) 场景1
3. 学习自动化脚本 - [README_SAFETY_AND_RESUME.md](README_SAFETY_AND_RESUME.md) 场景6

---

## 🔗 相关资源

- LeRobot官方文档: https://github.com/huggingface/lerobot
- CALVIN数据集: https://github.com/mees/calvin
- HuggingFace Hub: https://huggingface.co/

---

## 📝 更新日志

### v3.0 - 安全版（最新）
- ✅ 新增安全防护功能
- ✅ 新增断点续传功能
- ✅ 新增进度跟踪
- ✅ 优化分批处理

### v2.0 - 采样版
- ✅ 新增随机采样功能
- ✅ 支持可重复采样

### v1.0 - 基础版
- ✅ 内存优化
- ✅ 图像压缩
- ✅ 垃圾回收

---

## 🤝 获取帮助

如果遇到问题：

1. 查看相关文档
2. 检查错误日志
3. 尝试使用 `--resume` 恢复
4. 降低 `--image_quality` 或 `--writer_threads`

---

## ⭐ 推荐配置总结

**最安全的配置**（新手推荐）:
```bash
python convert_calvin_to_lerobot_safe.py \
    --data_dir /path/to/calvin \
    --create_backup \
    --resume
```

**最快的配置**（有经验用户）:
```bash
python convert_calvin_to_lerobot_safe.py \
    --data_dir /path/to/calvin \
    --force_delete \
    --writer_threads 4 \
    --image_quality 85
```

**最省内存的配置**:
```bash
python convert_calvin_to_lerobot_safe.py \
    --data_dir /path/to/calvin \
    --image_quality 75 \
    --writer_threads 1 \
    --batch_save_episodes 5
```

**最适合大数据集的配置**:
```bash
python convert_calvin_to_lerobot_safe.py \
    --data_dir /path/to/calvin \
    --max_episodes 5000 \
    --resume \
    --batch_save_episodes 10
```

---

**祝你使用愉快！** 🚀

有问题随时查看文档或重新运行 `--resume` 命令。