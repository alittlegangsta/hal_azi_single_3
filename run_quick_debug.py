#!/usr/bin/env python3
"""
快速调试脚本 - 使用极小数据量验证整个流程
专门用于快速测试，不适合实际分析
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
warnings.filterwarnings('ignore')

# 导入主要分析器
from main_analysis import CementChannelingAnalyzer

# 导入各个功能模块
from wellpath_alignment import add_alignment_to_analyzer, WellpathAlignment
from regression_target import add_regression_target_to_analyzer  
from wavelet_transform import add_wavelet_transform_to_analyzer, WaveletTransformProcessor

def run_quick_debug():
    """运行快速调试版本"""
    print("="*80)
    print("🚀 快速调试模式 - 极小数据量验证")
    print("数据量：5ft深度范围，预计100-200个样本")
    print("="*80)
    
    try:
        # ===============================
        # 第1步：初始化和数据准备
        # ===============================
        print("\n" + "="*60)
        print("第1步：数据注入与准备")
        print("="*60)
        
        analyzer = CementChannelingAnalyzer()
        
        # 添加功能模块
        add_alignment_to_analyzer()
        add_regression_target_to_analyzer()
        add_wavelet_transform_to_analyzer()
        
        # 加载数据
        analyzer.load_data()
        analyzer.structure_data()
        analyzer.preprocess_sonic_waveforms()
        
        print("✅ 第1步完成：数据注入与准备")
        
        # ===============================
        # 第2步：数据对齐（极小样本模式）
        # ===============================
        print("\n" + "="*60)
        print("第2步：高精度时空-方位数据对齐（极小样本模式）")
        print("="*60)
        
        # 设置极小样本深度范围
        tiny_sample_range = (2732, 2737)  # 只有5ft范围，约36个深度点
        print(f"🔧 极小样本深度范围: {tiny_sample_range[0]:.1f} - {tiny_sample_range[1]:.1f} ft")
        
        # 修改对齐器的默认深度范围
        original_init = WellpathAlignment.__init__
        def patched_init(self, analyzer):
            original_init(self, analyzer)
            self.unified_depth_range = tiny_sample_range
        WellpathAlignment.__init__ = patched_init
        
        analyzer.run_alignment_section()
        print("✅ 第2步完成：数据对齐")
        
        # ===============================
        # 第3步：CSI计算
        # ===============================
        print("\n" + "="*60)
        print("第3步：构建量化的回归目标（深度范围±0.25ft CSI）")
        print("="*60)
        
        analyzer.run_regression_target_section()
        
        # 验证CSI数据
        csi_data = analyzer.target_builder.csi_data
        print(f"📊 CSI样本数量: {len(csi_data)}")
        print(f"📈 CSI分布: {csi_data['csi'].min():.3f} - {csi_data['csi'].max():.3f}")
        print("✅ 第3步完成：CSI计算")
        
        # ===============================
        # 第4步：优化的小波变换
        # ===============================
        print("\n" + "="*60)
        print("第4步：连续小波变换时频分解（优化版本）")
        print("="*60)
        
        # 进一步优化小波变换参数
        def patched_wavelet_init(self, analyzer):
            """优化的小波变换参数"""
            self.analyzer = analyzer
            self.sampling_rate = 100000  # 100 kHz
            self.target_freq_range = (1000, 15000)  # 缩小频率范围：1-15 kHz
            self.wavelet_name = 'cmor1.5-1.0'
            self.scales = None
            self.frequencies = None
            self.scalograms_dataset = None
        
        # 临时替换初始化方法
        original_wavelet_init = WaveletTransformProcessor.__init__
        WaveletTransformProcessor.__init__ = patched_wavelet_init
        
        analyzer.run_wavelet_transform_section()
        
        # 恢复原始方法
        WaveletTransformProcessor.__init__ = original_wavelet_init
        
        # 验证小波数据
        scalograms = analyzer.wavelet_processor.scalograms_dataset
        print(f"📊 尺度图数据集形状: {scalograms['scalograms'].shape}")
        print(f"📈 频率范围: {scalograms['frequencies'].min():.1f} Hz - {scalograms['frequencies'].max()/1000:.1f} kHz")
        print("✅ 第4步完成：小波变换")
        
        # ===============================
        # 第5步：简化的模型训练模拟
        # ===============================
        print("\n" + "="*60)
        print("第5步：模型训练模拟（快速版本）")
        print("="*60)
        
        # 由于样本数量很少，只进行模拟训练
        n_samples = len(scalograms['csi_labels'])
        print(f"📊 总样本数: {n_samples}")
        
        if n_samples < 50:
            print("⚠️  样本数量过少，无法进行实际训练")
            print("📊 生成模拟训练结果...")
            
            # 创建模拟的训练历史图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            epochs = np.arange(1, 11)
            mock_loss = 0.2 * np.exp(-epochs/5) + np.random.normal(0, 0.01, len(epochs))
            mock_mae = 0.15 * np.exp(-epochs/6) + np.random.normal(0, 0.008, len(epochs))
            
            ax1.plot(epochs, mock_loss, 'b-', label='Training Loss (Mock)')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss (MSE)')
            ax1.set_title('Model Training History - Loss (Mock)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(epochs, mock_mae, 'r-', label='Training MAE (Mock)')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.set_title('Model Training History - MAE (Mock)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('quick_debug_training_mock.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("📊 模拟训练历史图已保存")
            
        else:
            print("✅ 样本数量足够，可以进行实际训练")
        
        print("✅ 第5步完成：模型训练模拟")
        
        # ===============================
        # 总结
        # ===============================
        print("\n" + "="*80)
        print("🎉 快速调试流程执行成功！")
        print("="*80)
        
        print("\n📋 快速调试总结:")
        print(f"  • 数据样本数: {len(csi_data)} 个")
        print(f"  • 深度范围: {tiny_sample_range[0]}-{tiny_sample_range[1]} ft")
        print(f"  • CSI范围: {csi_data['csi'].min():.3f}-{csi_data['csi'].max():.3f}")
        print(f"  • 尺度图形状: {scalograms['scalograms'].shape}")
        print(f"  • 处理时间: 预计<5分钟")
        
        print("\n📁 生成的文件:")
        debug_files = [
            "filtering_effect_comparison.png",
            "alignment_results.png", 
            "channeling_distribution.png",
            "csi_distribution_analysis.png",
            "wavelet_scales_design.png",
            "sample_scalograms.png",
            "quick_debug_training_mock.png",
        ]
        
        for filename in debug_files:
            if Path(filename).exists():
                file_size = Path(filename).stat().st_size / (1024*1024)
                print(f"  ✅ {filename} ({file_size:.1f} MB)")
            else:
                print(f"  ❌ {filename} (未生成)")
        
        print("\n🚀 快速调试完成！现在可以运行完整版本。")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 执行过程中出现错误: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        
        import traceback
        print("\n详细错误信息:")
        traceback.print_exc()
        
        return False

if __name__ == "__main__":
    success = run_quick_debug()
    
    if success:
        print("\n🎯 快速调试成功！")
        print("\n📋 下一步建议：")
        print("  1. 运行 run_complete_small_sample.py (优化后版本)")
        print("  2. 检查生成的可视化结果")
        print("  3. 根据需要调整参数")
    else:
        print("\n❌ 快速调试失败，请检查错误信息。") 