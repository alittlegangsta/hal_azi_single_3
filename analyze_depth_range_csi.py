#!/usr/bin/env python3
"""
分析深度范围CSI计算的验证脚本
验证±0.25ft深度范围的二维区域CSI计算是否正确
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import pickle
from pathlib import Path

def analyze_depth_range_csi():
    """分析深度范围CSI计算逻辑"""
    
    print("=== 深度范围CSI计算验证分析 ===\n")
    
    # 加载处理后的数据
    data_file = 'processed_data.pkl'
    if not Path(data_file).exists():
        print(f"错误：数据文件 {data_file} 不存在")
        print("请先运行完整的分析流程生成数据文件")
        return
    
    print(f"正在加载数据文件: {data_file}")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    print("数据加载完成！\n")
    
    # 分析新的CSI数据结构
    csi_data = data['csi_data']
    print("=== 新的CSI数据分析 ===")
    print(f"总样本数: {len(csi_data)}")
    print(f"深度范围: {csi_data['depth'].min():.1f} - {csi_data['depth'].max():.1f} ft")
    
    # 检查新增的字段
    if 'depth_range_size' in csi_data.columns:
        print(f"\n深度范围统计:")
        print(f"  平均深度点数: {csi_data['depth_points'].mean():.1f}")
        print(f"  平均方位点数: {csi_data['azimuth_points'].mean():.1f}")
        print(f"  平均区域总点数: {csi_data['region_total_points'].mean():.1f}")
        print(f"  深度范围大小分布: {csi_data['depth_range_size'].min()}-{csi_data['depth_range_size'].max()}")
        
        # 验证深度范围是否正确
        sample_idx = 0
        sample = csi_data.iloc[sample_idx]
        print(f"\n样本验证 (第{sample_idx+1}个样本):")
        print(f"  中心深度: {sample['depth']:.3f} ft")
        print(f"  深度范围: {sample['depth_range_min']:.3f} - {sample['depth_range_max']:.3f} ft")
        print(f"  深度范围宽度: {sample['depth_range_max'] - sample['depth_range_min']:.3f} ft")
        print(f"  深度点数: {sample['depth_points']}")
        print(f"  方位点数: {sample['azimuth_points']}")
        print(f"  区域总点数: {sample['region_total_points']}")
        print(f"  窜槽点数: {sample['region_channeling_points']}")
        print(f"  CSI: {sample['csi']:.4f}")
        
    # 比较CSI分布
    csi_values = csi_data['csi'].values
    print(f"\n=== CSI分布统计 ===")
    print(f"CSI范围: {csi_values.min():.4f} - {csi_values.max():.4f}")
    print(f"CSI均值: {csi_values.mean():.4f} ± {csi_values.std():.4f}")
    print(f"CSI中位数: {np.median(csi_values):.4f}")
    
    # 按接收器分析
    print(f"\n=== 按接收器分析 ===")
    for receiver in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
        receiver_data = csi_data[csi_data['receiver'] == receiver]
        if len(receiver_data) > 0:
            receiver_csi = receiver_data['csi'].values
            receiver_points = receiver_data['region_total_points'].values
            print(f"接收器 {receiver}: 样本数={len(receiver_data)}, "
                  f"CSI={receiver_csi.mean():.4f}±{receiver_csi.std():.4f}, "
                  f"平均区域点数={receiver_points.mean():.1f}")
    
    # 可视化分析
    visualize_depth_range_effect(csi_data)
    
    print("\n=== 深度范围CSI计算验证完成 ===")

def visualize_depth_range_effect(csi_data):
    """可视化深度范围效果"""
    print("\n正在生成深度范围效果可视化...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Depth Range CSI Calculation Analysis (±0.25ft)', fontsize=16)
    
    # 1. 深度点数分布
    ax = axes[0, 0]
    ax.hist(csi_data['depth_points'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xlabel('Depth Points per Region')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Depth Points\nin ±0.25ft Range')
    ax.grid(True, alpha=0.3)
    
    # 2. 方位点数分布
    ax = axes[0, 1]
    ax.hist(csi_data['azimuth_points'], bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
    ax.set_xlabel('Azimuth Points per Sector')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Azimuth Points\nin 45° Sector')
    ax.grid(True, alpha=0.3)
    
    # 3. 区域总点数分布
    ax = axes[0, 2]
    ax.hist(csi_data['region_total_points'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    ax.set_xlabel('Total Points per Region')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Total Points\n(Depth × Azimuth)')
    ax.grid(True, alpha=0.3)
    
    # 4. CSI vs 区域点数散点图
    ax = axes[1, 0]
    scatter = ax.scatter(csi_data['region_total_points'], csi_data['csi'], 
                        alpha=0.6, c=csi_data['depth_points'], cmap='viridis')
    ax.set_xlabel('Region Total Points')
    ax.set_ylabel('CSI')
    ax.set_title('CSI vs Region Size\n(Colored by depth points)')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Depth Points')
    
    # 5. 深度范围宽度分布
    depth_range_width = csi_data['depth_range_max'] - csi_data['depth_range_min']
    ax = axes[1, 1]
    ax.hist(depth_range_width, bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax.set_xlabel('Actual Depth Range Width (ft)')
    ax.set_ylabel('Frequency')
    ax.set_title('Actual Depth Range Distribution')
    ax.grid(True, alpha=0.3)
    
    # 添加统计信息
    ax.axvline(depth_range_width.mean(), color='red', linestyle='--', linewidth=2,
              label=f'Mean: {depth_range_width.mean():.3f} ft')
    ax.legend()
    
    # 6. CSI vs 深度（显示深度范围）
    ax = axes[1, 2]
    depths = csi_data['depth'].values
    csi_values = csi_data['csi'].values
    
    # 绘制CSI散点
    ax.scatter(depths, csi_values, alpha=0.6, s=20, c='blue', label='CSI values')
    
    # 绘制深度范围示例（选择几个点）
    sample_indices = np.linspace(0, len(csi_data)-1, 10, dtype=int)
    for i, idx in enumerate(sample_indices):
        row = csi_data.iloc[idx]
        depth_min = row['depth_range_min']
        depth_max = row['depth_range_max']
        csi = row['csi']
        
        # 绘制深度范围线
        ax.plot([depth_min, depth_max], [csi, csi], 'r-', alpha=0.5, linewidth=1)
        if i == 0:  # 只在第一个添加标签
            ax.plot([depth_min, depth_max], [csi, csi], 'r-', alpha=0.5, linewidth=1, 
                   label='±0.25ft range')
    
    ax.set_xlabel('Depth (ft)')
    ax.set_ylabel('CSI')
    ax.set_title('CSI vs Depth\n(Red lines show ±0.25ft ranges)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('depth_range_csi_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("深度范围效果图已保存为 depth_range_csi_analysis.png")

if __name__ == "__main__":
    analyze_depth_range_csi() 