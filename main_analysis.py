#!/usr/bin/env python3
"""
基于小波变换与梯度加权可解释性的声波测井水泥窜槽特征识别深度学习框架
"""

import numpy as np
import pandas as pd
import scipy.io
from scipy import signal
import matplotlib.pyplot as plt
import pywt
import h5py
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class CementChannelingAnalyzer:
    """声波测井水泥窜槽分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.ultrasonic_data = None
        self.sonic_data = None
        self.orientation_data = None
        self.processed_data = None
        self.unified_depth_axis = None
        
    def load_data(self):
        """第1.1节：多模态测井数据加载"""
        print("正在加载测井数据...")
        
        # 1.1.1 数据加载流程
        try:
            # 加载超声数据
            print("  加载超声数据 (CAST.mat)...")
            ultra_mat = scipy.io.loadmat('data/raw/CAST.mat')
            cast_struct = ultra_mat['CAST'][0, 0]  # 获取结构体
            self.ultrasonic_data = {
                'Depth': cast_struct['Depth'].flatten(),
                'Zc': cast_struct['Zc']  # 180 x 24750
            }
            
            # 加载声波数据 (仅使用三号阵列接收器)
            print("  加载声波数据 (XSILMR03.mat)...")
            sonic_mat = scipy.io.loadmat('data/raw/XSILMR/XSILMR03.mat')
            xsilmr_struct = sonic_mat['XSILMR03'][0, 0]  # 获取结构体
            self.sonic_data = {
                'Depth': xsilmr_struct['Depth'].flatten(),
                'WaveRng03SideA': xsilmr_struct['WaveRng03SideA'],  # 1024 x 7108
                'WaveRng03SideB': xsilmr_struct['WaveRng03SideB'],
                'WaveRng03SideC': xsilmr_struct['WaveRng03SideC'],
                'WaveRng03SideD': xsilmr_struct['WaveRng03SideD'],
                'WaveRng03SideE': xsilmr_struct['WaveRng03SideE'],
                'WaveRng03SideF': xsilmr_struct['WaveRng03SideF'],
                'WaveRng03SideG': xsilmr_struct['WaveRng03SideG'],
                'WaveRng03SideH': xsilmr_struct['WaveRng03SideH']
            }
            
            # 加载方位校正数据 (这个文件结构是直接的)
            print("  加载方位校正数据 (D2_XSI_RelBearing_Inclination.mat)...")
            orient_mat = scipy.io.loadmat('data/raw/D2_XSI_RelBearing_Inclination.mat')
            self.orientation_data = {
                'Depth_inc': orient_mat['Depth_inc'].flatten(),
                'Inc': orient_mat['Inc'].flatten(),  # 井斜角
                'RelBearing': orient_mat['RelBearing'].flatten()  # 相对方位角
            }
            
            print("  数据加载完成!")
            
        except Exception as e:
            print(f"数据加载错误: {e}")
            raise
        
        # 1.1.2 数据初步验证
        self._validate_data()
        
    def _validate_data(self):
        """数据初步验证"""
        print("正在验证数据完整性...")
        
        # 验证超声数据维度
        expected_zc_shape = (180, 24750)
        actual_zc_shape = self.ultrasonic_data['Zc'].shape
        if actual_zc_shape != expected_zc_shape:
            print(f"  警告: 超声Zc数据维度 {actual_zc_shape} 与预期 {expected_zc_shape} 不符")
        else:
            print(f"  ✓ 超声Zc数据维度验证通过: {actual_zc_shape}")
        
        # 验证声波数据维度 
        expected_wave_shape = (1024, 7108)
        for side in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            wave_key = f'WaveRng03Side{side}'
            actual_shape = self.sonic_data[wave_key].shape
            if actual_shape != expected_wave_shape:
                print(f"  警告: {wave_key}数据维度 {actual_shape} 与预期 {expected_wave_shape} 不符")
            else:
                print(f"  ✓ {wave_key}数据维度验证通过")
        
        # 数值范围检查
        ultra_depth_range = (self.ultrasonic_data['Depth'].min(), self.ultrasonic_data['Depth'].max())
        sonic_depth_range = (self.sonic_data['Depth'].min(), self.sonic_data['Depth'].max())
        orient_depth_range = (self.orientation_data['Depth_inc'].min(), self.orientation_data['Depth_inc'].max())
        
        print(f"  超声深度范围: {ultra_depth_range[0]:.1f} - {ultra_depth_range[1]:.1f} ft")
        print(f"  声波深度范围: {sonic_depth_range[0]:.1f} - {sonic_depth_range[1]:.1f} ft")
        print(f"  方位深度范围: {orient_depth_range[0]:.1f} - {orient_depth_range[1]:.1f} ft")
        
        # 检查角度范围
        inc_range = (self.orientation_data['Inc'].min(), self.orientation_data['Inc'].max())
        bearing_range = (self.orientation_data['RelBearing'].min(), self.orientation_data['RelBearing'].max())
        print(f"  井斜角范围: {inc_range[0]:.1f} - {inc_range[1]:.1f} 度")
        print(f"  相对方位角范围: {bearing_range[0]:.1f} - {bearing_range[1]:.1f} 度")
        
    def structure_data(self):
        """第1.2节：分析专用数据结构化"""
        print("正在进行数据结构化...")
        
        # 1.2.1 结构化方案 - 使用pandas DataFrame
        
        # 超声数据DataFrame
        ultrasonic_df = pd.DataFrame({
            'Depth': self.ultrasonic_data['Depth']
        })
        
        # 方位校正数据DataFrame
        orientation_df = pd.DataFrame({
            'Depth': self.orientation_data['Depth_inc'],
            'Inclination': self.orientation_data['Inc'],
            'RelativeBearing': self.orientation_data['RelBearing']
        })
        
        # 声波数据使用字典组织
        sonic_sides = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        
        # 创建时间轴 (1024个样点，采样间隔10μs)
        time_axis = np.arange(1024) * 10e-6  # 10微秒 = 10e-6秒
        
        # 组织声波数据为结构化字典
        sonic_structured = {
            'time_axis': time_axis,
            'depth': self.sonic_data['Depth'],
            'azimuthal_receivers': np.arange(8) * 45,  # 8个接收器，间隔45度
            'waveforms': {}
        }
        
        for side in sonic_sides:
            wave_key = f'WaveRng03Side{side}'
            sonic_structured['waveforms'][side] = self.sonic_data[wave_key]
        
        self.structured_data = {
            'ultrasonic': ultrasonic_df,
            'orientation': orientation_df,
            'sonic': sonic_structured
        }
        
        print("  数据结构化完成!")
        print(f"  - 超声数据: {len(ultrasonic_df)} 个深度点")
        print(f"  - 方位数据: {len(orientation_df)} 个深度点")
        print(f"  - 声波数据: {len(sonic_structured['depth'])} 个深度点, 8个方位接收器")
        
    def preprocess_sonic_waveforms(self):
        """第1.3节：声波波形预处理"""
        print("正在进行声波波形预处理...")
        
        # 1.3.1 高通滤波设计
        sampling_rate = 1 / (10e-6)  # 100 kHz采样率
        nyquist_freq = sampling_rate / 2
        cutoff_freq = 1000  # 1000 Hz截止频率
        
        # 设计4阶Butterworth高通滤波器
        sos = signal.butter(4, cutoff_freq / nyquist_freq, btype='high', output='sos')
        
        print(f"  高通滤波器参数:")
        print(f"  - 类型: 4阶Butterworth")
        print(f"  - 截止频率: {cutoff_freq} Hz")
        print(f"  - 采样率: {sampling_rate/1000:.0f} kHz")
        
        # 保存原始数据副本用于对比可视化
        self.original_sonic_data = {}
        for side in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            wave_key = f'WaveRng03Side{side}'
            self.original_sonic_data[wave_key] = self.sonic_data[wave_key].copy()
        
        # 对所有接收器的波形进行零相位滤波
        sonic_sides = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        filtered_sonic_data = {}
        
        for side in sonic_sides:
            wave_key = f'WaveRng03Side{side}'
            original_waves = self.sonic_data[wave_key]
            
            # 对每个深度点的波形进行滤波
            filtered_waves = np.zeros_like(original_waves)
            for depth_idx in range(original_waves.shape[1]):
                waveform = original_waves[:, depth_idx]
                # 使用filtfilt进行零相位滤波
                filtered_waveform = signal.sosfiltfilt(sos, waveform)
                filtered_waves[:, depth_idx] = filtered_waveform
            
            filtered_sonic_data[wave_key] = filtered_waves
            
        # 更新sonic_data
        self.sonic_data.update(filtered_sonic_data)
        
        print("  高通滤波完成!")
        
        # 1.3.2 预处理后可视化
        self._visualize_filtering_effect()
        
    def _visualize_filtering_effect(self):
        """可视化滤波效果"""
        print("正在生成滤波效果对比图...")
        
        # 随机选择几个深度点进行对比
        np.random.seed(42)
        depth_indices = np.random.choice(self.sonic_data['WaveRng03SideA'].shape[1], 3, replace=False)
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle('Sonic Waveform Filtering Effect Comparison', fontsize=16)
        
        # 时间轴 (转换为毫秒)
        time_ms = np.arange(1024) * 10e-3  # 10微秒转换为毫秒
        
        for i, depth_idx in enumerate(depth_indices):
            depth_value = self.sonic_data['Depth'][depth_idx]
            
            # 使用保存的原始数据副本
            original_wave = self.original_sonic_data['WaveRng03SideA'][:, depth_idx]
            filtered_wave = self.sonic_data['WaveRng03SideA'][:, depth_idx]
            
            # 绘制原始波形
            axes[i, 0].plot(time_ms, original_wave, 'b-', linewidth=1)
            axes[i, 0].set_title(f'Original Waveform (Depth: {depth_value:.1f} ft)')
            axes[i, 0].set_xlabel('Time (ms)')
            axes[i, 0].set_ylabel('Amplitude')
            axes[i, 0].grid(True, alpha=0.3)
            
            # 绘制滤波后波形
            axes[i, 1].plot(time_ms, filtered_wave, 'r-', linewidth=1)
            axes[i, 1].set_title(f'Filtered Waveform (>1000Hz)')
            axes[i, 1].set_xlabel('Time (ms)')
            axes[i, 1].set_ylabel('Amplitude')
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('filtering_effect_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("  滤波效果对比图已保存为 filtering_effect_comparison.png")


def main():
    """主函数"""
    print("="*60)
    print("基于小波变换与梯度加权可解释性的声波测井水泥窜槽特征识别深度学习框架")
    print("="*60)
    
    # 创建分析器实例
    analyzer = CementChannelingAnalyzer()
    
    # 第1节：基础数据注入与准备
    print("\n第1节：基础数据注入与准备")
    print("-"*40)
    
    try:
        # 1.1 多模态测井数据加载
        analyzer.load_data()
        
        # 1.2 分析专用数据结构化
        analyzer.structure_data()
        
        # 1.3 声波波形预处理
        analyzer.preprocess_sonic_waveforms()
        
        print("\n第1节完成！")
        
    except Exception as e:
        print(f"第1节执行失败: {e}")
        raise


if __name__ == "__main__":
    main() 