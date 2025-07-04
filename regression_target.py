#!/usr/bin/env python3
"""
第3节：构建量化的回归目标模块
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class RegressionTargetBuilder:
    """回归目标构建器"""
    
    def __init__(self, analyzer):
        """初始化回归目标构建器"""
        self.analyzer = analyzer
        self.channeling_threshold = 2.5  # 窜槽识别阈值
        self.channeling_map = None
        self.csi_dataset = None
        
    def identify_channeling_from_ultrasonic(self):
        """第3.1节：从超声数据中识别窜槽"""
        print("正在从超声数据中识别窜槽...")
        
        # 获取对齐后的超声数据
        aligned_data = self.analyzer.aligned_data
        ultrasonic_zc = aligned_data['ultrasonic']['Zc']  # 180 x n_depths
        depths = aligned_data['ultrasonic']['Depth']
        
        print(f"  超声数据形状: {ultrasonic_zc.shape}")
        print(f"  窜槽阈值: Zc < {self.channeling_threshold}")
        
        # 3.1 阈值法应用
        # 如果Zc值小于2.5，标记为窜槽(1)；否则标记为胶结良好(0)
        channeling_binary_map = (ultrasonic_zc < self.channeling_threshold).astype(int)
        
        # 统计窜槽分布
        total_points = ultrasonic_zc.size
        valid_points = ~np.isnan(ultrasonic_zc)
        channeling_points = np.sum(channeling_binary_map[valid_points])
        channeling_percentage = (channeling_points / np.sum(valid_points)) * 100
        
        print(f"  总数据点: {total_points:,}")
        print(f"  有效数据点: {np.sum(valid_points):,}")
        print(f"  窜槽点数: {channeling_points:,}")
        print(f"  窜槽比例: {channeling_percentage:.2f}%")
        
        # 存储高分辨率胶结图
        self.channeling_map = {
            'binary_map': channeling_binary_map,
            'depths': depths,
            'azimuths': np.arange(0, 360, 2),  # 每2度一个方位角
            'threshold': self.channeling_threshold,
            'statistics': {
                'total_points': total_points,
                'valid_points': np.sum(valid_points),
                'channeling_points': channeling_points,
                'channeling_percentage': channeling_percentage
            }
        }
        
        print("  窜槽识别完成！")
        
        # 可视化窜槽分布
        self._visualize_channeling_map()
        
    def _visualize_channeling_map(self):
        """可视化窜槽分布图"""
        print("  正在生成窜槽分布可视化...")
        
        depths = self.channeling_map['depths']
        azimuths = self.channeling_map['azimuths']
        binary_map = self.channeling_map['binary_map']
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # 1. 窜槽分布热力图
        ax = axes[0]
        im = ax.imshow(binary_map, aspect='auto', cmap='RdYlBu_r', 
                      extent=[depths.min(), depths.max(), azimuths.min(), azimuths.max()],
                      origin='lower')
        ax.set_xlabel('Depth (ft)')
        ax.set_ylabel('Azimuth (degree)')
        ax.set_title('Channeling Distribution Map\n(Red: Channeling, Blue: Good Bond)')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Channeling (0=Good, 1=Poor)')
        
        # 2. 深度方向的窜槽比例曲线
        ax = axes[1]
        depth_channeling_ratio = np.nanmean(binary_map, axis=0)
        ax.plot(depths, depth_channeling_ratio * 100, 'r-', linewidth=2)
        ax.set_xlabel('Depth (ft)')
        ax.set_ylabel('Channeling Percentage (%)')
        ax.set_title('Channeling Percentage vs Depth')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        # 添加统计信息
        stats = self.channeling_map['statistics']
        textstr = f"Total Channeling: {stats['channeling_percentage']:.1f}%\n"
        textstr += f"Threshold: Zc < {self.channeling_threshold}"
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('channeling_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("  窜槽分布图已保存为 channeling_distribution.png")
    
    def compute_channeling_severity_index(self):
        """第3.2节：推导连续的窜槽严重性指数(CSI)"""
        print("正在计算窜槽严重性指数(CSI)...")
        
        # 获取扇区匹配结果
        if not hasattr(self.analyzer, 'aligner') or not hasattr(self.analyzer.aligner, 'matched_sectors'):
            raise ValueError("必须先完成第2节的方位对齐才能计算CSI")
        
        matched_sectors = self.analyzer.aligner.matched_sectors
        binary_map = self.channeling_map['binary_map']
        depths = self.channeling_map['depths']
        azimuths = self.channeling_map['azimuths']
        
        print(f"  匹配扇区数量: {len(matched_sectors)}")
        
        # 深度范围参数
        depth_range = 0.25  # ±0.25ft的深度范围
        print(f"  使用深度范围: ±{depth_range} ft")
        
        # 计算每个声波波形的CSI
        csi_data = []
        
        for idx, row in matched_sectors.iterrows():
            depth_idx = row['depth_index']
            current_depth = row['depth']
            receiver = row['receiver']
            receiver_azimuth = row['receiver_azimuth']
            sector_zc_values = row['sector_zc_values']
            
            # 计算深度范围
            depth_min = current_depth - depth_range
            depth_max = current_depth + depth_range
            
            # 找到深度范围内的所有索引
            depth_range_mask = (depths >= depth_min) & (depths <= depth_max)
            depth_range_indices = np.where(depth_range_mask)[0]
            
            # 根据接收器方位角找到对应的超声数据扇区
            sector_half_width = 22.5
            sector_min = (receiver_azimuth - sector_half_width) % 360
            sector_max = (receiver_azimuth + sector_half_width) % 360
            
            # 找到该扇区内的超声方位角索引
            if sector_min < sector_max:
                sector_mask = (azimuths >= sector_min) & (azimuths <= sector_max)
            else:
                sector_mask = (azimuths >= sector_min) | (azimuths <= sector_max)
            
            sector_azimuth_indices = np.where(sector_mask)[0]
            
            if len(sector_azimuth_indices) > 0 and len(depth_range_indices) > 0:
                # 获取二维区域内的窜槽标识
                # 使用meshgrid获取所有(方位,深度)组合
                azi_mesh, depth_mesh = np.meshgrid(sector_azimuth_indices, depth_range_indices, indexing='ij')
                
                # 提取二维区域内的所有窜槽标识
                region_channeling_flags = binary_map[azi_mesh.flatten(), depth_mesh.flatten()]
                
                # 过滤掉NaN值
                valid_flags = region_channeling_flags[~np.isnan(region_channeling_flags)]
                
                if len(valid_flags) > 0:
                    # 计算CSI：二维区域内窜槽点数量 / 总点数量
                    csi = np.sum(valid_flags) / len(valid_flags)
                    
                    # 存储结果
                    csi_data.append({
                        'depth': current_depth,
                        'depth_index': depth_idx,
                        'depth_range_size': len(depth_range_indices),
                        'depth_range_min': depths[depth_range_indices].min(),
                        'depth_range_max': depths[depth_range_indices].max(),
                        'receiver': receiver,
                        'receiver_index': row['receiver_index'],
                        'receiver_azimuth': receiver_azimuth,
                        'csi': csi,
                        'region_total_points': len(valid_flags),
                        'region_channeling_points': int(np.sum(valid_flags)),
                        'azimuth_points': len(sector_azimuth_indices),
                        'depth_points': len(depth_range_indices)
                    })
        
        # 转换为DataFrame
        self.csi_data = pd.DataFrame(csi_data)
        
        print(f"  CSI计算完成，共 {len(self.csi_data)} 个样本")
        
        # 统计深度范围覆盖情况
        if len(self.csi_data) > 0:
            avg_depth_points = self.csi_data['depth_points'].mean()
            avg_azimuth_points = self.csi_data['azimuth_points'].mean()
            avg_total_points = self.csi_data['region_total_points'].mean()
            
            print(f"  深度范围统计:")
            print(f"    平均深度点数: {avg_depth_points:.1f}")
            print(f"    平均方位点数: {avg_azimuth_points:.1f}")
            print(f"    平均区域总点数: {avg_total_points:.1f}")
        
        # 统计CSI分布
        csi_values = self.csi_data['csi'].values
        print(f"  CSI统计:")
        print(f"    范围: {csi_values.min():.3f} - {csi_values.max():.3f}")
        print(f"    均值: {csi_values.mean():.3f} ± {csi_values.std():.3f}")
        print(f"    中位数: {np.median(csi_values):.3f}")
        
        # 分析不同CSI等级的分布
        csi_categories = {
            'Excellent (CSI < 0.1)': np.sum(csi_values < 0.1),
            'Good (0.1 ≤ CSI < 0.3)': np.sum((csi_values >= 0.1) & (csi_values < 0.3)),
            'Fair (0.3 ≤ CSI < 0.6)': np.sum((csi_values >= 0.3) & (csi_values < 0.6)),
            'Poor (CSI ≥ 0.6)': np.sum(csi_values >= 0.6)
        }
        
        print("  CSI分布统计:")
        for category, count in csi_categories.items():
            percentage = (count / len(csi_values)) * 100
            print(f"    {category}: {count} ({percentage:.1f}%)")
        
        # 可视化CSI分布
        self._visualize_csi_distribution()
        
    def _visualize_csi_distribution(self):
        """可视化CSI分布"""
        print("  正在生成CSI分布可视化...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Channeling Severity Index (CSI) Distribution Analysis\n(Computed over ±0.25ft depth range)', fontsize=16)
        
        csi_values = self.csi_data['csi'].values
        depths = self.csi_data['depth'].values
        receivers = self.csi_data['receiver'].values
        region_points = self.csi_data['region_total_points'].values
        
        # 1. CSI直方图
        ax = axes[0, 0]
        ax.hist(csi_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Channeling Severity Index (CSI)')
        ax.set_ylabel('Frequency')
        ax.set_title('CSI Distribution Histogram\n(±0.25ft depth range)')
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        ax.axvline(csi_values.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {csi_values.mean():.3f}')
        ax.axvline(np.median(csi_values), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(csi_values):.3f}')
        ax.legend()
        
        # 2. CSI vs 深度散点图，用区域点数着色
        ax = axes[0, 1]
        scatter = ax.scatter(depths, csi_values, c=region_points, cmap='viridis', alpha=0.6, s=20)
        ax.set_xlabel('Depth (ft)')
        ax.set_ylabel('CSI')
        ax.set_title('CSI vs Depth\n(Colored by region points)')
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Region Total Points')
        
        # 3. 不同接收器的CSI箱线图
        ax = axes[1, 0]
        receiver_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        csi_by_receiver = [csi_values[receivers == receiver] for receiver in receiver_list]
        
        box_plot = ax.boxplot(csi_by_receiver, labels=receiver_list, patch_artist=True)
        ax.set_xlabel('Receiver')
        ax.set_ylabel('CSI')
        ax.set_title('CSI Distribution by Receiver\n(±0.25ft depth range)')
        ax.grid(True, alpha=0.3)
        
        # 给箱线图添加颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(box_plot['boxes'])))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        # 4. 区域点数分布直方图
        ax = axes[1, 1]
        ax.hist(region_points, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        ax.set_xlabel('Region Total Points')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Region Points\n(Depth range × Azimuth sector)')
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        ax.axvline(region_points.mean(), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {region_points.mean():.1f}')
        ax.axvline(np.median(region_points), color='green', linestyle='--', linewidth=2, 
                  label=f'Median: {np.median(region_points):.1f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('csi_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("  CSI分布分析图已保存为 csi_distribution_analysis.png")
    
    def prepare_model_dataset(self):
        """准备用于模型训练的数据集"""
        print("正在准备模型训练数据集...")
        
        # 获取对齐后的声波数据
        aligned_sonic = self.analyzer.aligned_data['sonic']
        
        dataset_samples = []
        
        for idx, row in self.csi_data.iterrows():
            depth_idx = row['depth_index']
            receiver = row['receiver']
            csi = row['csi']
            
            # 获取对应的声波波形
            wave_key = f'WaveRng03Side{receiver}'
            waveform = aligned_sonic[wave_key][:, depth_idx]
            
            # 检查波形是否有效
            if not np.any(np.isnan(waveform)):
                dataset_samples.append({
                    'waveform': waveform,
                    'csi': csi,
                    'depth': row['depth'],
                    'receiver': receiver,
                    'depth_index': depth_idx,
                    'receiver_index': row['receiver_index']
                })
        
        print(f"  准备了 {len(dataset_samples)} 个有效的(波形,CSI)对")
        
        # 创建最终数据集
        waveforms = np.array([sample['waveform'] for sample in dataset_samples])
        csi_labels = np.array([sample['csi'] for sample in dataset_samples])
        
        self.model_dataset = {
            'waveforms': waveforms,
            'csi_labels': csi_labels,
            'metadata': pd.DataFrame([{k: v for k, v in sample.items() if k not in ['waveform']} 
                                    for sample in dataset_samples])
        }
        
        print(f"  数据集形状:")
        print(f"    波形: {waveforms.shape}")
        print(f"    CSI标签: {csi_labels.shape}")
        print(f"    标签范围: {csi_labels.min():.3f} - {csi_labels.max():.3f}")
        
    def save_processed_data(self, filepath='processed_data.h5'):
        """第3.3节：数据持久化"""
        print(f"正在保存处理后的数据到 {filepath}...")
        
        with h5py.File(filepath, 'w') as f:
            # 保存模型数据集
            f.create_dataset('waveforms', data=self.model_dataset['waveforms'])
            f.create_dataset('csi_labels', data=self.model_dataset['csi_labels'])
            
            # 保存元数据
            metadata_df = self.model_dataset['metadata']
            for col in metadata_df.columns:
                f.create_dataset(f'metadata/{col}', data=metadata_df[col].values)
            
            # 保存配置信息
            f.attrs['channeling_threshold'] = self.channeling_threshold
            f.attrs['unified_depth_range'] = self.analyzer.aligner.unified_depth_range
            f.attrs['depth_resolution'] = self.analyzer.aligner.depth_resolution
            f.attrs['n_samples'] = len(self.model_dataset['csi_labels'])
            
            # 保存统计信息
            stats = self.channeling_map['statistics']
            for key, value in stats.items():
                f.attrs[f'channeling_stats_{key}'] = value
        
        # 同时保存为pickle文件，便于快速加载
        pickle_data = {
            'model_dataset': self.model_dataset,
            'csi_data': self.csi_data,
            'channeling_map': self.channeling_map,
            'config': {
                'channeling_threshold': self.channeling_threshold,
                'unified_depth_range': self.analyzer.aligner.unified_depth_range,
                'depth_resolution': self.analyzer.aligner.depth_resolution
            }
        }
        
        pickle_filepath = filepath.replace('.h5', '.pkl')
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(pickle_data, f)
        
        print(f"  数据保存完成!")
        print(f"    HDF5文件: {filepath}")
        print(f"    Pickle文件: {pickle_filepath}")
        
    @staticmethod
    def load_processed_data(filepath='processed_data.h5'):
        """加载处理后的数据"""
        pickle_filepath = filepath.replace('.h5', '.pkl')
        
        if Path(pickle_filepath).exists():
            print(f"从 {pickle_filepath} 加载数据...")
            with open(pickle_filepath, 'rb') as f:
                return pickle.load(f)
        elif Path(filepath).exists():
            print(f"从 {filepath} 加载数据...")
            data = {}
            with h5py.File(filepath, 'r') as f:
                data['model_dataset'] = {
                    'waveforms': f['waveforms'][:],
                    'csi_labels': f['csi_labels'][:]
                }
                # 加载其他数据...
            return data
        else:
            raise FileNotFoundError(f"未找到数据文件: {filepath} 或 {pickle_filepath}")


def add_regression_target_to_analyzer():
    """将回归目标构建功能添加到主分析器"""
    
    def run_regression_target_section(self):
        """运行第3节：构建量化的回归目标"""
        print("\n第3节：构建量化的回归目标")
        print("-"*40)
        
        # 创建回归目标构建器
        target_builder = RegressionTargetBuilder(self)
        
        try:
            # 3.1 从超声数据中识别窜槽
            target_builder.identify_channeling_from_ultrasonic()
            
            # 3.2 推导连续的窜槽严重性指数
            target_builder.compute_channeling_severity_index()
            
            # 准备模型数据集
            target_builder.prepare_model_dataset()
            
            # 3.3 数据持久化
            target_builder.save_processed_data()
            
            # 将target_builder存储到analyzer中供后续使用
            self.target_builder = target_builder
            
            print("\n第3节完成！")
            
        except Exception as e:
            print(f"第3节执行失败: {e}")
            raise
    
    # 将方法动态添加到CementChannelingAnalyzer类
    from main_analysis import CementChannelingAnalyzer
    CementChannelingAnalyzer.run_regression_target_section = run_regression_target_section 