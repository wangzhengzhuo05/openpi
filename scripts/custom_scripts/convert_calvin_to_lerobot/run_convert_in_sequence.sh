#!/bin/bash
set -e  # 当任意命令出错时立即退出

echo "🚀 开始执行第一阶段: 前 13000 episodes"
uv run /root/autodl-tmp/openpi/scripts/custom_scripts/convert_calvin_to_lerobot/convert_calvin_to_lerobot.py --max_episodes 13000 --clean_start

echo "✅ 第一阶段完成，准备进入第二阶段..."

uv run /root/autodl-tmp/openpi/scripts/custom_scripts/convert_calvin_to_lerobot/convert_calvin_to_lerobot.py --resume --max_episodes 13000
echo "🎉 全部执行完成！"
