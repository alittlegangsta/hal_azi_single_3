#!/usr/bin/env python3
"""
第4节：通过连续小波变换进行时频分解模块
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

class WaveletTransformProcessor:
    """连续小波变换处理器"""
    
    def __init__(self, analyzer):
        """初始化小波变换处理器"""
        self.analyzer = analyzer
        self.sampling_rate = 100000  # 100 kHz (10微秒采样间隔)
        self.target_freq_range = (1, 30000)  # 1 Hz 到 30 kHz
        self.wavelet_name = 'cmor1.5-1.0'  # 复数Morlet小波
        self.scales = None
        self.frequencies = None
        self.scalograms_dataset = None
        
    def design_wavelet_scales(self):
        """设计小波尺度以覆盖目标频率范围"""
        print("正在设计小波尺度...")
        
        # 采样周期
        sampling_period = 1.0 / self.sampling_rate
        
        # 目标频率范围
        freq_min, freq_max = self.target_freq_range
        
        print(f"  采样率: {self.sampling_rate/1000:.0f} kHz")
        print(f"  目标频率范围: {freq_min} Hz - {freq_max/1000:.0f} kHz")
        print(f"  选择的小波: {self.wavelet_name}")
        
        # 计算对应的尺度范围
        # 使用pywt.scale2frequency函数进行转换
        # f = pywt.scale2frequency(wavelet, scale) / sampling_period
        
        # 从最高频率到最低频率，对数间隔生成尺度
        # 更高的频率对应更小的尺度
        scale_max = pywt.scale2frequency(self.wavelet_name, 1) / (freq_min * sampling_period)
        scale_min = pywt.scale2frequency(self.wavelet_name, 1) / (freq_max * sampling_period)
        
        # 生成对数间隔的尺度数组
        n_scales = 30  # 减少到30个尺度以提高计算速度
        self.scales = np.logspace(np.log10(scale_min), np.log10(scale_max), n_scales)
        
        # 计算对应的频率
        self.frequencies = pywt.scale2frequency(self.wavelet_name, self.scales) / sampling_period
        
        print(f"  生成了 {len(self.scales)} 个尺度")
        print(f"  实际频率范围: {self.frequencies.min():.1f} Hz - {self.frequencies.max()/1000:.1f} kHz")
        
        # 可视化尺度-频率关系
        self._visualize_scale_frequency_relationship()
        
    def _visualize_scale_frequency_relationship(self):
        """可视化尺度与频率的关系"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. 尺度分布
        ax1.semilogx(self.scales, np.arange(len(self.scales)), 'b.-')
        ax1.set_xlabel('Scale')
        ax1.set_ylabel('Scale Index')
        ax1.set_title('Wavelet Scales Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 2. 频率分布
        ax2.semilogx(self.frequencies, np.arange(len(self.frequencies)), 'r.-')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Scale Index')
        ax2.set_title('Corresponding Frequencies Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('wavelet_scales_design.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("  尺度设计图已保存为 wavelet_scales_design.png")
    
    def apply_cwt_to_dataset(self):
        """第4.3节：对整个数据集应用CWT生成尺度图"""
        print("正在对数据集应用连续小波变换...")
        
        # 获取模型数据集
        if not hasattr(self.analyzer, 'target_builder'):
            raise ValueError("必须先完成第3节的回归目标构建才能进行CWT")
        
        waveforms = self.analyzer.target_builder.model_dataset['waveforms']
        csi_labels = self.analyzer.target_builder.model_dataset['csi_labels']
        metadata = self.analyzer.target_builder.model_dataset['metadata']
        
        print(f"  处理 {len(waveforms)} 个波形...")
        print(f"  波形长度: {waveforms.shape[1]} 个样点")
        
        # 初始化尺度图数组
        n_samples = len(waveforms)
        n_scales = len(self.scales)
        n_time_samples = waveforms.shape[1]
        
        scalograms = np.zeros((n_samples, n_scales, n_time_samples), dtype=np.float32)
        
        # 批量处理波形
        print("  正在进行小波变换...")
        for i, waveform in enumerate(waveforms):
            if i % 50 == 0:  # 每50个样本显示一次进度，更频繁的更新
                print(f"    处理进度: {i+1}/{n_samples} ({(i+1)/n_samples*100:.1f}%)")
            
            # 应用连续小波变换
            cwt_coefficients, _ = pywt.cwt(waveform, self.scales, self.wavelet_name)
            
            # 计算尺度图（取复数系数的模）
            scalogram = np.abs(cwt_coefficients)
            scalograms[i] = scalogram.astype(np.float32)
        
        print("  小波变换完成！")
        
        # 创建尺度图数据集
        self.scalograms_dataset = {
            'scalograms': scalograms,
            'csi_labels': csi_labels,
            'metadata': metadata,
            'scales': self.scales,
            'frequencies': self.frequencies,
            'time_axis': np.arange(n_time_samples) * (1.0 / self.sampling_rate),  # 时间轴（秒）
            'transform_params': {
                'wavelet': self.wavelet_name,
                'sampling_rate': self.sampling_rate,
                'freq_range': self.target_freq_range,
                'n_scales': n_scales
            }
        }
        
        print(f"  尺度图数据集形状: {scalograms.shape}")
        print(f"  时间轴范围: 0 - {self.scalograms_dataset['time_axis'][-1]*1000:.1f} ms")
        print(f"  频率轴范围: {self.frequencies.min():.1f} Hz - {self.frequencies.max()/1000:.1f} kHz")
        
        # 可视化几个样本的尺度图
        self._visualize_sample_scalograms()
        
    def _visualize_sample_scalograms(self):
        """可视化几个样本的尺度图"""
        print("  正在生成样本尺度图可视化...")
        
        scalograms = self.scalograms_dataset['scalograms']
        csi_labels = self.scalograms_dataset['csi_labels']
        time_axis_ms = self.scalograms_dataset['time_axis'] * 1000  # 转换为毫秒
        frequencies_khz = self.frequencies / 1000  # 转换为kHz
        
        # 选择不同CSI等级的样本
        low_csi_idx = np.argmin(csi_labels)  # 最好的胶结
        high_csi_idx = np.argmax(csi_labels)  # 最差的胶结
        medium_csi_idx = np.argmin(np.abs(csi_labels - np.median(csi_labels)))  # 中等胶结
        
        sample_indices = [low_csi_idx, medium_csi_idx, high_csi_idx]
        sample_titles = [
            f'Good Bond (CSI={csi_labels[low_csi_idx]:.3f})',
            f'Medium Bond (CSI={csi_labels[medium_csi_idx]:.3f})',
            f'Poor Bond (CSI={csi_labels[high_csi_idx]:.3f})'
        ]
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Sample Scalograms from CWT Analysis', fontsize=16)
        
        for i, (idx, title) in enumerate(zip(sample_indices, sample_titles)):
            # 原始波形
            ax = axes[i, 0]
            original_waveform = self.analyzer.target_builder.model_dataset['waveforms'][idx]
            ax.plot(time_axis_ms, original_waveform, 'b-', linewidth=1)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Original Waveform - {title}')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 4)  # 聚焦于0-4 ms
            
            # 尺度图
            ax = axes[i, 1]
            scalogram = scalograms[idx]
            
            # 限制显示范围到0-4ms和30kHz以下
            time_mask = time_axis_ms <= 4
            freq_mask = frequencies_khz <= 30
            
            display_scalogram = scalogram[freq_mask, :][:, time_mask]
            display_time = time_axis_ms[time_mask]
            display_freq = frequencies_khz[freq_mask]
            
            im = ax.imshow(display_scalogram, aspect='auto', cmap='jet',
                          extent=[display_time[0], display_time[-1], 
                                 display_freq[0], display_freq[-1]],
                          origin='lower')
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Frequency (kHz)')
            ax.set_title(f'Scalogram - {title}')
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Magnitude')
        
        plt.tight_layout()
        plt.savefig('sample_scalograms.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("  样本尺度图已保存为 sample_scalograms.png")
    
    def analyze_scalogram_statistics(self):
        """分析尺度图的统计特性"""
        print("正在分析尺度图统计特性...")
        
        scalograms = self.scalograms_dataset['scalograms']
        csi_labels = self.scalograms_dataset['csi_labels']
        
        # 计算全局统计
        print(f"  尺度图统计:")
        print(f"    形状: {scalograms.shape}")
        print(f"    数据类型: {scalograms.dtype}")
        print(f"    值范围: {scalograms.min():.3f} - {scalograms.max():.3f}")
        print(f"    均值: {scalograms.mean():.3f} ± {scalograms.std():.3f}")
        
        # 按CSI等级分组分析
        csi_thresholds = [0.1, 0.3, 0.6]
        csi_groups = {
            'Excellent': csi_labels < csi_thresholds[0],
            'Good': (csi_labels >= csi_thresholds[0]) & (csi_labels < csi_thresholds[1]),
            'Fair': (csi_labels >= csi_thresholds[1]) & (csi_labels < csi_thresholds[2]),
            'Poor': csi_labels >= csi_thresholds[2]
        }
        
        print("\n  按胶结质量分组的尺度图特性:")
        for group_name, group_mask in csi_groups.items():
            if np.any(group_mask):
                group_scalograms = scalograms[group_mask]
                print(f"    {group_name}: {np.sum(group_mask)} 个样本")
                print(f"      均值: {group_scalograms.mean():.3f}")
                print(f"      标准差: {group_scalograms.std():.3f}")
                print(f"      最大值: {group_scalograms.max():.3f}")
        
        # 分析时频域能量分布
        self._analyze_time_frequency_energy_distribution()
        
    def _analyze_time_frequency_energy_distribution(self):
        """分析时频域能量分布"""
        print("  分析时频域能量分布...")
        
        scalograms = self.scalograms_dataset['scalograms']
        csi_labels = self.scalograms_dataset['csi_labels']
        time_axis_ms = self.scalograms_dataset['time_axis'] * 1000
        frequencies_khz = self.frequencies / 1000
        
        # 计算不同CSI等级的平均尺度图
        csi_low = csi_labels <= 0.2  # 低CSI（好胶结）
        csi_high = csi_labels >= 0.6  # 高CSI（差胶结）
        
        if np.any(csi_low) and np.any(csi_high):
            avg_scalogram_low = np.mean(scalograms[csi_low], axis=0)
            avg_scalogram_high = np.mean(scalograms[csi_high], axis=0)
            
            # 计算差异图
            scalogram_diff = avg_scalogram_high - avg_scalogram_low
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Time-Frequency Energy Distribution Analysis', fontsize=16)
            
            # 限制显示范围
            time_mask = time_axis_ms <= 4
            freq_mask = frequencies_khz <= 30
            
            display_time = time_axis_ms[time_mask]
            display_freq = frequencies_khz[freq_mask]
            
            # 好胶结平均尺度图
            ax = axes[0, 0]
            im = ax.imshow(avg_scalogram_low[freq_mask, :][:, time_mask], 
                          aspect='auto', cmap='jet',
                          extent=[display_time[0], display_time[-1], 
                                 display_freq[0], display_freq[-1]],
                          origin='lower')
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Frequency (kHz)')
            ax.set_title(f'Good Bond Average (CSI ≤ 0.2, n={np.sum(csi_low)})')
            plt.colorbar(im, ax=ax)
            
            # 差胶结平均尺度图
            ax = axes[0, 1]
            im = ax.imshow(avg_scalogram_high[freq_mask, :][:, time_mask], 
                          aspect='auto', cmap='jet',
                          extent=[display_time[0], display_time[-1], 
                                 display_freq[0], display_freq[-1]],
                          origin='lower')
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Frequency (kHz)')
            ax.set_title(f'Poor Bond Average (CSI ≥ 0.6, n={np.sum(csi_high)})')
            plt.colorbar(im, ax=ax)
            
            # 差异图
            ax = axes[1, 0]
            im = ax.imshow(scalogram_diff[freq_mask, :][:, time_mask], 
                          aspect='auto', cmap='RdBu_r',
                          extent=[display_time[0], display_time[-1], 
                                 display_freq[0], display_freq[-1]],
                          origin='lower')
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Frequency (kHz)')
            ax.set_title('Difference (Poor - Good)')
            plt.colorbar(im, ax=ax, label='Magnitude Difference')
            
            # 时间和频率维度的平均能量
            ax = axes[1, 1]
            
            # 时间维度平均（所有频率）
            time_avg_low = np.mean(avg_scalogram_low, axis=0)
            time_avg_high = np.mean(avg_scalogram_high, axis=0)
            
            ax.plot(time_axis_ms, time_avg_low, 'b-', linewidth=2, label='Good Bond', alpha=0.7)
            ax.plot(time_axis_ms, time_avg_high, 'r-', linewidth=2, label='Poor Bond', alpha=0.7)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Average Magnitude')
            ax.set_title('Time Domain Average Energy')
            ax.set_xlim(0, 4)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('time_frequency_energy_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("  时频能量分析图已保存为 time_frequency_energy_analysis.png")
    
    def save_scalogram_dataset(self, filepath='scalogram_dataset.npz'):
        """保存尺度图数据集"""
        print(f"正在保存尺度图数据集到 {filepath}...")
        
        # 保存为压缩的npz文件
        np.savez_compressed(
            filepath,
            scalograms=self.scalograms_dataset['scalograms'],
            csi_labels=self.scalograms_dataset['csi_labels'],
            scales=self.scalograms_dataset['scales'],
            frequencies=self.scalograms_dataset['frequencies'],
            time_axis=self.scalograms_dataset['time_axis'],
            metadata_depth=self.scalograms_dataset['metadata']['depth'].values,
            metadata_receiver=self.scalograms_dataset['metadata']['receiver'].values,
            metadata_receiver_index=self.scalograms_dataset['metadata']['receiver_index'].values,
            **self.scalograms_dataset['transform_params']
        )
        
        print(f"  尺度图数据集保存完成: {filepath}")
        
        # 打印文件大小信息
        import os
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  文件大小: {file_size_mb:.1f} MB")
    
    @staticmethod
    def load_scalogram_dataset(filepath='scalogram_dataset.npz'):
        """加载尺度图数据集"""
        print(f"从 {filepath} 加载尺度图数据集...")
        
        data = np.load(filepath)
        
        scalogram_dataset = {
            'scalograms': data['scalograms'],
            'csi_labels': data['csi_labels'],
            'scales': data['scales'],
            'frequencies': data['frequencies'],
            'time_axis': data['time_axis'],
            'metadata': {
                'depth': data['metadata_depth'],
                'receiver': data['metadata_receiver'],
                'receiver_index': data['metadata_receiver_index']
            },
            'transform_params': {
                'wavelet': str(data['wavelet']),
                'sampling_rate': float(data['sampling_rate']),
                'freq_range': tuple(data['freq_range']),
                'n_scales': int(data['n_scales'])
            }
        }
        
        print(f"  加载完成，尺度图形状: {scalogram_dataset['scalograms'].shape}")
        return scalogram_dataset


def add_wavelet_transform_to_analyzer():
    """将小波变换功能添加到主分析器"""
    
    def run_wavelet_transform_section(self):
        """运行第4节：通过连续小波变换进行时频分解"""
        print("\n第4节：通过连续小波变换进行时频分解")
        print("-"*40)
        
        # 创建小波变换处理器
        wavelet_processor = WaveletTransformProcessor(self)
        
        try:
            # 4.1 & 4.2 设计小波尺度
            wavelet_processor.design_wavelet_scales()
            
            # 4.3 应用CWT生成尺度图数据集
            wavelet_processor.apply_cwt_to_dataset()
            
            # 分析尺度图统计特性
            wavelet_processor.analyze_scalogram_statistics()
            
            # 保存尺度图数据集
            wavelet_processor.save_scalogram_dataset()
            
            # 将wavelet_processor存储到analyzer中供后续使用
            self.wavelet_processor = wavelet_processor
            
            print("\n第4节完成！")
            
        except Exception as e:
            print(f"第4节执行失败: {e}")
            raise
    
    # 将方法动态添加到CementChannelingAnalyzer类
    from main_analysis import CementChannelingAnalyzer
    CementChannelingAnalyzer.run_wavelet_transform_section = run_wavelet_transform_section 