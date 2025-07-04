#!/usr/bin/env python3
"""
基于小波变换与梯度加权可解释性的声波测井水泥窜槽特征识别深度学习框架
完整分析流程运行脚本
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# 导入主要分析器
from main_analysis import CementChannelingAnalyzer

# 导入各个功能模块
from wellpath_alignment import add_alignment_to_analyzer
from regression_target import add_regression_target_to_analyzer  
from wavelet_transform import add_wavelet_transform_to_analyzer

def install_dependencies():
    """检查并安装必要的依赖包"""
    try:
        import subprocess
        import sys
        
        print("检查依赖包...")
        
        # 检查关键包是否已安装
        required_packages = [
            'numpy', 'pandas', 'scipy', 'matplotlib', 
            'sklearn', 'h5py', 'xarray', 'pywt'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"缺少依赖包: {missing_packages}")
            print("正在安装依赖包...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("依赖包安装完成!")
        else:
            print("所有依赖包已就绪!")
            
    except Exception as e:
        print(f"依赖包检查失败: {e}")
        print("请手动运行: pip install -r requirements.txt")

def check_data_files():
    """检查数据文件是否存在"""
    required_files = [
        'data/raw/CAST.mat',
        'data/raw/XSILMR/XSILMR03.mat',
        'data/raw/D2_XSI_RelBearing_Inclination.mat'
    ]
    
    print("检查数据文件...")
    missing_files = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            file_size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"  ✓ {file_path} ({file_size:.1f} MB)")
    
    if missing_files:
        print(f"❌ 缺少数据文件: {missing_files}")
        return False
    
    print("所有数据文件检查完成!")
    return True

def run_section_with_error_handling(analyzer, section_func, section_name):
    """带错误处理的运行某个分析section"""
    try:
        print(f"\n{'='*60}")
        print(f"开始执行 {section_name}")
        print(f"{'='*60}")
        
        section_func()
        
        print(f"\n✅ {section_name} 执行成功!")
        return True
        
    except Exception as e:
        print(f"\n❌ {section_name} 执行失败: {str(e)}")
        print(f"错误详情: {type(e).__name__}")
        
        # 询问用户是否继续
        while True:
            choice = input(f"\n是否跳过 {section_name} 继续执行后续步骤? (y/n): ").strip().lower()
            if choice in ['y', 'yes', '是']:
                print(f"跳过 {section_name}，继续执行...")
                return False
            elif choice in ['n', 'no', '否']:
                print("停止执行。")
                sys.exit(1)
            else:
                print("请输入 y(继续) 或 n(停止)")

def main():
    """主函数"""
    print("="*80)
    print("基于小波变换与梯度加权可解释性的声波测井水泥窜槽特征识别深度学习框架")
    print("="*80)
    print("版本: 1.0")
    print("作者: AI 助手")
    print("="*80)
    
    # 1. 安装依赖包
    install_dependencies()
    
    # 2. 检查数据文件
    if not check_data_files():
        print("请确保所有数据文件都存在后再运行程序。")
        sys.exit(1)
    
    # 3. 创建分析器并添加功能模块
    print("\n初始化分析器...")
    analyzer = CementChannelingAnalyzer()
    
    # 动态添加各个功能模块
    add_alignment_to_analyzer()
    add_regression_target_to_analyzer()
    add_wavelet_transform_to_analyzer()
    
    print("分析器初始化完成!")
    
    # 4. 执行各个分析步骤
    analysis_steps = [
        (analyzer.load_data, "数据加载"),
        (analyzer.structure_data, "数据结构化"),
        (analyzer.preprocess_sonic_waveforms, "声波预处理"),
        (analyzer.run_alignment_section, "第2节: 高精度时空-方位数据对齐"),
        (analyzer.run_regression_target_section, "第3节: 构建量化的回归目标"),
        (analyzer.run_wavelet_transform_section, "第4节: 连续小波变换时频分解")
    ]
    
    successful_steps = []
    
    for step_func, step_name in analysis_steps:
        success = run_section_with_error_handling(analyzer, step_func, step_name)
        if success:
            successful_steps.append(step_name)
    
    # 5. 总结报告
    print("\n" + "="*80)
    print("分析完成总结")
    print("="*80)
    
    print(f"成功完成的步骤 ({len(successful_steps)}/{len(analysis_steps)}):")
    for i, step in enumerate(successful_steps, 1):
        print(f"  {i}. ✅ {step}")
    
    failed_steps = len(analysis_steps) - len(successful_steps)
    if failed_steps > 0:
        print(f"\n跳过的步骤: {failed_steps} 个")
    
    # 6. 输出文件说明
    print("\n生成的输出文件:")
    output_files = [
        "filtering_effect_comparison.png - 滤波效果对比图",
        "alignment_results.png - 数据对齐结果可视化",
        "channeling_distribution.png - 窜槽分布图",
        "csi_distribution_analysis.png - CSI分布分析图",
        "wavelet_scales_design.png - 小波尺度设计图",
        "sample_scalograms.png - 样本尺度图",
        "time_frequency_energy_analysis.png - 时频能量分析图",
        "processed_data.h5 / processed_data.pkl - 处理后的数据",
        "scalogram_dataset.npz - 尺度图数据集"
    ]
    
    existing_files = []
    for file_desc in output_files:
        filename = file_desc.split(' - ')[0]
        if os.path.exists(filename):
            file_size = os.path.getsize(filename) / (1024*1024)  # MB
            existing_files.append(f"  ✓ {file_desc} ({file_size:.1f} MB)")
        else:
            existing_files.append(f"  ❌ {file_desc} (未生成)")
    
    for file_info in existing_files:
        print(file_info)
    
    print("\n" + "="*80)
    if len(successful_steps) == len(analysis_steps):
        print("🎉 所有分析步骤执行成功！")
        print("数据预处理和特征提取阶段已完成。")
        print("接下来可以进行模型训练（第5节）和可解释性分析（第6-7节）。")
    else:
        print("⚠️  部分步骤未能完成，请检查错误信息并重新运行。")
    
    print("="*80)

def run_quick_test():
    """快速测试模式 - 只处理少量数据"""
    print("运行快速测试模式...")
    
    # 可以在这里实现一个快速测试版本
    # 比如只处理前100个样本等
    pass

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_quick_test()
    else:
        main() 