#!/bin/bash

# 自动训练脚本 - 完整数据集
# 自动选择：1（重新开始）-> 4（完整数据集）

echo "🚀 开始自动训练脚本..."
echo "选择：1（重新开始）-> 4（完整数据集）"

# 使用here document自动输入选项
python run_complete_optimized.py << EOF
1
4
EOF

echo "✅ 训练脚本执行完成！" 