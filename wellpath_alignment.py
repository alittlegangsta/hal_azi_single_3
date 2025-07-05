#!/usr/bin/env python3
"""
第2节：高精度时空-方位数据对齐模块
"""

import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class WellpathAlignment:
    """井眼轨迹重建与数据对齐类"""
    
    def __init__(self, analyzer):
        """初始化对齐器"""
        self.analyzer = analyzer
        self.wellpath_data = None
        self.unified_depth_range = (2732, 4132)  # 目标研究深度范围 (ft)
        self.depth_resolution = 0.14  # 统一深度步长 (ft)
        
    def reconstruct_wellpath(self):
        """第2.1节：井眼轨迹重建 - 使用最小曲率法"""
        print("正在重建井眼轨迹...")
        
        # 获取方位数据
        orientation_data = self.analyzer.orientation_data
        
        # 实现最小曲率法
        depths = orientation_data['Depth_inc']
        inclinations = orientation_data['Inc']
        azimuths = orientation_data['RelBearing']
        
        # 初始化坐标
        n_points = len(depths)
        tvd = np.zeros(n_points)  # 真垂直深度
        north = np.zeros(n_points)  # 北向位移
        east = np.zeros(n_points)  # 东向位移
        
        # 最小曲率法计算
        for i in range(1, n_points):
            # 深度差
            delta_md = depths[i] - depths[i-1]
            
            # 角度转换为弧度
            inc1_rad = np.radians(inclinations[i-1])
            inc2_rad = np.radians(inclinations[i])
            azi1_rad = np.radians(azimuths[i-1])
            azi2_rad = np.radians(azimuths[i])
            
            # 计算曲率角度
            cos_beta = (np.cos(inc1_rad) * np.cos(inc2_rad) + 
                       np.sin(inc1_rad) * np.sin(inc2_rad) * 
                       np.cos(azi2_rad - azi1_rad))
            
            # 防止数值误差
            cos_beta = np.clip(cos_beta, -1, 1)
            beta = np.arccos(cos_beta)
            
            # 曲率因子
            if beta < 1e-6:  # 几乎直线
                rf = 1.0
            else:
                rf = 2 / beta * np.tan(beta / 2)
            
            # 计算坐标增量
            delta_tvd = 0.5 * delta_md * (np.cos(inc1_rad) + np.cos(inc2_rad)) * rf
            delta_north = 0.5 * delta_md * (np.sin(inc1_rad) * np.cos(azi1_rad) + 
                                           np.sin(inc2_rad) * np.cos(azi2_rad)) * rf
            delta_east = 0.5 * delta_md * (np.sin(inc1_rad) * np.sin(azi1_rad) + 
                                          np.sin(inc2_rad) * np.sin(azi2_rad)) * rf
            
            # 累积坐标
            tvd[i] = tvd[i-1] + delta_tvd
            north[i] = north[i-1] + delta_north
            east[i] = east[i-1] + delta_east
        
        # 存储井眼轨迹数据
        self.wellpath_data = pd.DataFrame({
            'Depth': depths,
            'TVD': tvd,
            'North': north,
            'East': east,
            'Inclination': inclinations,
            'Azimuth': azimuths
        })
        
        print(f"  井眼轨迹重建完成，共 {len(self.wellpath_data)} 个测点")
        print(f"  深度范围: {depths.min():.1f} - {depths.max():.1f} ft")
        print(f"  TVD范围: {tvd.min():.1f} - {tvd.max():.1f} ft")
        print(f"  水平位移: 北向{north.max()-north.min():.1f} ft, 东向{east.max()-east.min():.1f} ft")
        
    def create_unified_depth_axis(self):
        """第2.2节：统一深度基准注册"""
        print("正在创建统一深度基准...")
        
        # 定义目标深度范围内的统一深度轴
        start_depth, end_depth = self.unified_depth_range
        self.unified_depth_axis = np.arange(start_depth, end_depth + self.depth_resolution, 
                                          self.depth_resolution)
        
        print(f"  统一深度轴: {start_depth} - {end_depth} ft")
        print(f"  深度步长: {self.depth_resolution} ft")
        print(f"  总点数: {len(self.unified_depth_axis)}")
        
        # 对所有数据进行深度对齐
        self._interpolate_to_unified_depth()
        
    def _interpolate_to_unified_depth(self):
        """将所有数据插值到统一深度轴"""
        print("  正在进行深度插值对齐...")
        
        aligned_data = {}
        
        # 1. 插值超声数据
        ultra_depths = self.analyzer.ultrasonic_data['Depth']
        ultra_zc = self.analyzer.ultrasonic_data['Zc']
        
        # 为每个方位角创建插值函数
        aligned_zc = np.zeros((180, len(self.unified_depth_axis)))
        
        for azimuth_idx in range(180):
            # 线性插值
            interp_func = interpolate.interp1d(ultra_depths, ultra_zc[azimuth_idx, :], 
                                             kind='linear', bounds_error=False, 
                                             fill_value=np.nan)
            aligned_zc[azimuth_idx, :] = interp_func(self.unified_depth_axis)
        
        aligned_data['ultrasonic'] = {
            'Depth': self.unified_depth_axis,
            'Zc': aligned_zc
        }
        
        # 2. 插值声波数据
        sonic_depths = self.analyzer.sonic_data['Depth']
        aligned_sonic = {}
        
        for side in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            wave_key = f'WaveRng03Side{side}'
            wave_data = self.analyzer.sonic_data[wave_key]
            
            # 为每个时间样点创建插值函数
            aligned_wave = np.zeros((wave_data.shape[0], len(self.unified_depth_axis)))
            
            for time_idx in range(wave_data.shape[0]):
                interp_func = interpolate.interp1d(sonic_depths, wave_data[time_idx, :],
                                                 kind='linear', bounds_error=False,
                                                 fill_value=np.nan)
                aligned_wave[time_idx, :] = interp_func(self.unified_depth_axis)
            
            aligned_sonic[wave_key] = aligned_wave
        
        aligned_data['sonic'] = aligned_sonic
        aligned_data['sonic']['Depth'] = self.unified_depth_axis
        
        # 3. 插值方位校正数据
        orient_depths = self.analyzer.orientation_data['Depth_inc']
        orient_inc = self.analyzer.orientation_data['Inc']
        orient_bearing = self.analyzer.orientation_data['RelBearing']
        
        # 插值井斜角和相对方位角
        inc_interp = interpolate.interp1d(orient_depths, orient_inc, 
                                        kind='linear', bounds_error=False, fill_value=np.nan)
        bearing_interp = interpolate.interp1d(orient_depths, orient_bearing,
                                            kind='linear', bounds_error=False, fill_value=np.nan)
        
        aligned_inclination = inc_interp(self.unified_depth_axis)
        aligned_rel_bearing = bearing_interp(self.unified_depth_axis)
        
        aligned_data['orientation'] = {
            'Depth': self.unified_depth_axis,
            'Inclination': aligned_inclination,
            'RelativeBearing': aligned_rel_bearing
        }
        
        # 存储对齐后的数据
        self.analyzer.aligned_data = aligned_data
        
        print("  深度插值对齐完成!")
        
    def perform_azimuthal_correction(self):
        """第2.3节：方位校正与扇区匹配"""
        print("正在进行方位校正与扇区匹配...")
        
        # 获取对齐后的数据
        aligned_data = self.analyzer.aligned_data
        depths = aligned_data['orientation']['Depth']
        inclinations = aligned_data['orientation']['Inclination']
        rel_bearings = aligned_data['orientation']['RelativeBearing']
        
        # 2.3.1 绝对方位计算流程
        n_depths = len(depths)
        receiver_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        n_receivers = len(receiver_names)
        
        # 存储每个接收器在每个深度的绝对方位角
        absolute_azimuths = np.zeros((n_depths, n_receivers))
        
        # 假设井眼高边方位为0度 (可根据实际情况调整)
        wellbore_high_side_azimuth = 0.0
        
        for depth_idx in range(n_depths):
            # 获取当前深度的相对方位角
            rel_bearing = rel_bearings[depth_idx]
            
            if np.isnan(rel_bearing):
                absolute_azimuths[depth_idx, :] = np.nan
                continue
                
            # 计算A接收器的绝对方位角
            azimuth_A = (wellbore_high_side_azimuth + rel_bearing) % 360
            
            # 计算所有接收器的绝对方位角 (每个接收器间隔45度)
            for receiver_idx in range(n_receivers):
                absolute_azimuths[depth_idx, receiver_idx] = (azimuth_A + receiver_idx * 45) % 360
        
        # 存储绝对方位角数据
        self.absolute_azimuths = absolute_azimuths
        
        print(f"  绝对方位角计算完成，形状: {absolute_azimuths.shape}")
        
        # 2.3.2 扇区匹配
        self._perform_sector_matching()
        
    def _perform_sector_matching(self):
        """执行声波接收器与超声数据的扇区匹配"""
        print("  正在执行扇区匹配...")
        
        aligned_data = self.analyzer.aligned_data
        ultrasonic_zc = aligned_data['ultrasonic']['Zc']  # 180 x n_depths
        n_depths = len(aligned_data['ultrasonic']['Depth'])
        
        # 超声数据的方位角 (每2度一个，共180个)
        ultrasonic_azimuths = np.arange(0, 360, 2)  # 0, 2, 4, ..., 358
        
        # 为每个声波接收器定义扇区 (±22.5度)
        sector_half_width = 22.5
        
        # 存储匹配结果
        matched_data = []
        
        receiver_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        
        for depth_idx in range(n_depths):
            depth = aligned_data['ultrasonic']['Depth'][depth_idx]
            
            for receiver_idx, receiver_name in enumerate(receiver_names):
                # 获取该接收器在该深度的绝对方位角
                receiver_azimuth = self.absolute_azimuths[depth_idx, receiver_idx]
                
                if np.isnan(receiver_azimuth):
                    continue
                
                # 定义扇区范围
                sector_min = (receiver_azimuth - sector_half_width) % 360
                sector_max = (receiver_azimuth + sector_half_width) % 360
                
                # 找到落在该扇区内的超声测量点
                if sector_min < sector_max:
                    # 正常情况：扇区不跨越0度
                    sector_mask = (ultrasonic_azimuths >= sector_min) & (ultrasonic_azimuths <= sector_max)
                else:
                    # 跨越0度的情况
                    sector_mask = (ultrasonic_azimuths >= sector_min) | (ultrasonic_azimuths <= sector_max)
                
                # 获取该扇区内的超声数据
                sector_indices = np.where(sector_mask)[0]
                
                if len(sector_indices) > 0:
                    sector_zc_values = ultrasonic_zc[sector_indices, depth_idx]
                    
                    # 过滤掉NaN值
                    valid_zc_values = sector_zc_values[~np.isnan(sector_zc_values)]
                    
                    if len(valid_zc_values) > 0:
                        # 存储匹配信息
                        matched_data.append({
                            'depth': depth,
                            'depth_index': depth_idx,
                            'receiver': receiver_name,
                            'receiver_index': receiver_idx,
                            'receiver_azimuth': receiver_azimuth,
                            'sector_zc_values': valid_zc_values,
                            'n_sector_points': len(valid_zc_values)
                        })
        
        # 转换为DataFrame便于后续处理
        self.matched_sectors = pd.DataFrame(matched_data)
        
        print(f"  扇区匹配完成，共匹配 {len(self.matched_sectors)} 个声波-超声对应关系")
        
    def visualize_alignment_results(self):
        """可视化对齐结果"""
        print("正在生成对齐结果可视化图...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Data Alignment Results Visualization', fontsize=16)
        
        # 1. 井眼轨迹3D可视化
        if self.wellpath_data is not None:
            ax = axes[0, 0]
            ax.plot(self.wellpath_data['East'], self.wellpath_data['North'], 'b-', linewidth=2)
            ax.set_xlabel('East (ft)')
            ax.set_ylabel('North (ft)')
            ax.set_title('Wellbore Trajectory (Plan View)')
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
        
        # 2. 深度对齐效果
        ax = axes[0, 1]
        depths = self.analyzer.aligned_data['ultrasonic']['Depth']
        sample_zc = self.analyzer.aligned_data['ultrasonic']['Zc'][0, :]  # 第一个方位的数据
        
        ax.plot(depths, sample_zc, 'g-', linewidth=1)
        ax.set_xlabel('Depth (ft)')
        ax.set_ylabel('Acoustic Impedance')
        ax.set_title('Aligned Ultrasonic Data (0° Azimuth)')
        ax.grid(True, alpha=0.3)
        
        # 3. 方位校正结果
        ax = axes[1, 0]
        if hasattr(self, 'absolute_azimuths'):
            # 选择几个深度点展示方位校正
            depth_indices = np.linspace(0, len(depths)-1, 50, dtype=int)
            selected_depths = depths[depth_indices]
            selected_azimuths = self.absolute_azimuths[depth_indices, :]
            
            for receiver_idx in range(8):
                ax.plot(selected_depths, selected_azimuths[:, receiver_idx], 
                       'o-', markersize=3, linewidth=1, 
                       label=f'Receiver {chr(65+receiver_idx)}')
            
            ax.set_xlabel('Depth (ft)')
            ax.set_ylabel('Absolute Azimuth (degree)')
            ax.set_title('Azimuthal Correction Results')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # 4. 扇区匹配统计
        ax = axes[1, 1]
        if hasattr(self, 'matched_sectors'):
            sector_counts = self.matched_sectors.groupby('receiver')['n_sector_points'].mean()
            receivers = sector_counts.index
            counts = sector_counts.values
            
            bars = ax.bar(receivers, counts, alpha=0.7, color='skyblue')
            ax.set_xlabel('Receiver')
            ax.set_ylabel('Average Sector Points')
            ax.set_title('Sector Matching Statistics')
            ax.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{count:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('alignment_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("  对齐结果可视化图已保存为 alignment_results.png")


def add_alignment_to_analyzer():
    """将对齐功能添加到主分析器"""
    
    def run_alignment_section(self):
        """运行第2节：高精度时空-方位数据对齐"""
        print("\n第2节：高精度时空-方位数据对齐")
        print("-"*40)
        
        # 创建对齐器
        aligner = WellpathAlignment(self)
        
        try:
            # 2.1 井眼轨迹重建
            aligner.reconstruct_wellpath()
            
            # 2.2 统一深度基准注册
            aligner.create_unified_depth_axis()
            
            # 2.3 方位校正与扇区匹配
            aligner.perform_azimuthal_correction()
            
            # 可视化结果
            aligner.visualize_alignment_results()
            
            # 将aligner存储到analyzer中供后续使用
            self.aligner = aligner
            
            print("\n第2节完成！")
            
        except Exception as e:
            print(f"第2节执行失败: {e}")
            raise
    
    # 将方法动态添加到CementChannelingAnalyzer类
    from main_analysis import CementChannelingAnalyzer
    CementChannelingAnalyzer.run_alignment_section = run_alignment_section 