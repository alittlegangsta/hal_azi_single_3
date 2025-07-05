#!/bin/bash

# 自动训练脚本 - 快速验证
# 自动选择：1（重新开始）-> 5（超快速验证2K样本）

echo "🚀 开始快速验证训练脚本..."
echo "选择：1（重新开始）-> 5（超快速验证 2,000样本）"

# 使用here document自动输入选项
python run_complete_optimized.py << EOF
1
5
EOF

echo "✅ 快速验证训练脚本执行完成！" 