#!/usr/bin/env python3
"""
分析深度范围和CSI计算逻辑的脚本
"""

import pickle
import numpy as np
import pandas as pd

def analyze_depth_range():
    """分析深度范围和CSI计算逻辑"""
    
    print("正在加载处理后的数据...")
    
    # 加载处理后的数据
    with open('processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print("="*60)
    print("数据结构分析")
    print("="*60)
    
    # 分析配置信息
    config = data['config']
    print(f"配置信息:")
    print(f"  窜槽阈值: {config['channeling_threshold']}")
    print(f"  统一深度范围: {config['unified_depth_range'][0]:.1f} - {config['unified_depth_range'][1]:.1f} ft")
    print(f"  深度分辨率: {config['depth_resolution']} ft")
    
    # 计算统一深度轴
    start_depth, end_depth = config['unified_depth_range']
    depth_resolution = config['depth_resolution']
    unified_depth_axis = np.arange(start_depth, end_depth + depth_resolution, depth_resolution)
    
    print(f"\n统一深度轴:")
    print(f"  总深度点数: {len(unified_depth_axis)}")
    print(f"  深度范围: {unified_depth_axis[0]:.1f} - {unified_depth_axis[-1]:.1f} ft")
    print(f"  深度步长: {depth_resolution} ft")
    
    # 分析CSI数据
    csi_data = data['csi_data']
    print(f"\nCSI数据分析:")
    print(f"  总样本数: {len(csi_data)}")
    print(f"  深度范围: {csi_data['depth'].min():.1f} - {csi_data['depth'].max():.1f} ft")
    
    # 分析每个接收器的样本分布
    print(f"\n各接收器样本分布:")
    for receiver in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
        count = len(csi_data[csi_data['receiver'] == receiver])
        print(f"  接收器{receiver}: {count} 个样本")
    
    # 分析方位扇区匹配逻辑
    print(f"\n="*60)
    print("方位扇区匹配逻辑")
    print("="*60)
    
    print(f"超声数据方位角:")
    ultrasonic_azimuths = np.arange(0, 360, 2)  # 每2度一个
    print(f"  方位角数量: {len(ultrasonic_azimuths)}")
    print(f"  方位角范围: {ultrasonic_azimuths[0]}° - {ultrasonic_azimuths[-1]}°")
    print(f"  方位角间隔: 2°")
    
    print(f"\n声波接收器扇区:")
    sector_half_width = 22.5  # ±22.5度
    print(f"  扇区半宽度: ±{sector_half_width}°")
    print(f"  扇区全宽度: {2*sector_half_width}°")
    
    # 计算每个扇区包含的超声测量点数
    sector_points = []
    for i in range(8):
        receiver_azimuth = i * 45  # 接收器间隔45度
        sector_min = (receiver_azimuth - sector_half_width) % 360
        sector_max = (receiver_azimuth + sector_half_width) % 360
        
        if sector_min < sector_max:
            sector_mask = (ultrasonic_azimuths >= sector_min) & (ultrasonic_azimuths <= sector_max)
        else:
            sector_mask = (ultrasonic_azimuths >= sector_min) | (ultrasonic_azimuths <= sector_max)
        
        points_in_sector = np.sum(sector_mask)
        sector_points.append(points_in_sector)
        
        print(f"  接收器{chr(65+i)}({receiver_azimuth}°): 扇区[{sector_min:.1f}°, {sector_max:.1f}°], {points_in_sector}个超声点")
    
    print(f"\n每个扇区平均包含 {np.mean(sector_points):.1f} 个超声测量点")
    
    # 关键回答用户问题
    print(f"\n" + "="*60)
    print("回答用户问题：深度范围")
    print("="*60)
    
    print(f"对于每一个深度点的声波波形，其对应的超声数据窜槽比例计算:")
    print(f"\n1. 深度维度:")
    print(f"   - 并非基于深度范围，而是基于精确的深度点匹配")
    print(f"   - 声波和超声数据都插值到统一深度轴上")
    print(f"   - 每个深度点使用该深度点的数据，无深度范围概念")
    print(f"   - 深度点间隔: {depth_resolution} ft")
    
    print(f"\n2. 方位维度:")
    print(f"   - 每个声波接收器对应一个方位扇区 (±{sector_half_width}°)")
    print(f"   - 在该扇区内的所有超声测量点参与窜槽比例计算")
    print(f"   - 扇区内平均包含 {np.mean(sector_points):.0f} 个超声点")
    
    print(f"\n3. CSI计算公式:")
    print(f"   CSI = 扇区内窜槽点数量 / 扇区内总点数量")
    print(f"   其中窜槽点定义为: Zc < {config['channeling_threshold']}")
    
    print(f"\n4. 总结:")
    print(f"   每个声波波形对应的是：")
    print(f"   - 特定深度点: 深度点间隔{depth_resolution} ft")
    print(f"   - 特定方位扇区: 宽度{2*sector_half_width}°，包含~{int(np.mean(sector_points))}个超声点")
    print(f"   - 没有深度范围的概念，是点对点的精确匹配")

if __name__ == "__main__":
    analyze_depth_range() 