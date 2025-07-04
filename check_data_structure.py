#!/usr/bin/env python3
"""
检查MATLAB文件结构的脚本
"""

import scipy.io
import numpy as np

def check_mat_file(filepath):
    """检查MATLAB文件的结构"""
    print(f"\n检查文件: {filepath}")
    print("="*50)
    
    try:
        data = scipy.io.loadmat(filepath)
        
        # 过滤掉MATLAB的元数据
        filtered_keys = [k for k in data.keys() if not k.startswith('__')]
        
        print(f"文件中的变量 ({len(filtered_keys)} 个):")
        for key in filtered_keys:
            value = data[key]
            if isinstance(value, np.ndarray):
                print(f"  {key}: {value.shape}, 类型: {value.dtype}")
                if value.size < 20:  # 如果数据量小，显示实际值
                    print(f"    值: {value.flatten()[:10]}")
                else:
                    print(f"    范围: {value.min():.3f} - {value.max():.3f}")
            else:
                print(f"  {key}: {type(value)}, 值: {value}")
                
    except Exception as e:
        print(f"读取文件失败: {e}")

def main():
    """主函数"""
    
    # 检查所有数据文件
    files_to_check = [
        'data/raw/CAST.mat',
        'data/raw/XSILMR/XSILMR03.mat',
        'data/raw/D2_XSI_RelBearing_Inclination.mat'
    ]
    
    for filepath in files_to_check:
        check_mat_file(filepath)

if __name__ == "__main__":
    main() 