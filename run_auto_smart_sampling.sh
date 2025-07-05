#!/bin/bash

# 自动训练脚本 - 智能采样（推荐）
# 自动选择：1（重新开始）-> 1（智能采样10K样本）

echo "🚀 开始智能采样训练脚本..."
echo "选择：1（重新开始）-> 1（智能采样 10,000样本）"

# 使用here document自动输入选项
python run_complete_optimized.py << EOF
1
1
EOF

echo "✅ 智能采样训练脚本执行完成！" 