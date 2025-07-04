# 基于小波变换与梯度加权可解释性的声波测井水泥窜槽特征识别深度学习框架

## 项目概述

本项目实现了一个完整的端到端深度学习框架，用于从声波测井数据中自动识别水泥窜槽特征。该框架结合了连续小波变换(CWT)和梯度加权类激活映射(Grad-CAM)技术，不仅能够准确预测窜槽严重性，还能提供可解释的决策依据。

## 主要特性

- **多模态数据融合**: 整合声波、超声和方位校正数据
- **精确的时空对齐**: 基于最小曲率法的井眼轨迹重建和动态方位校正
- **连续回归目标**: 引入窜槽严重性指数(CSI)作为量化评估指标
- **高级时频分析**: 采用复数Morlet小波进行多分辨率时频分解
- **深度学习预测**: 定制化CNN模型进行回归预测
- **可解释性分析**: 回归适应性Grad-CAM揭示模型决策机制
- **信号重构验证**: 通过逆小波变换实现可逆性分析

## 项目结构

```
hal_azi_single_3/
├── data/
│   └── raw/
│       ├── CAST.mat                                    # 超声数据
│       ├── D2_XSI_RelBearing_Inclination.mat          # 方位校正数据
│       └── XSILMR/
│           └── XSILMR03.mat                           # 声波数据
├── main_analysis.py                                   # 主分析器 (第1节)
├── wellpath_alignment.py                              # 数据对齐模块 (第2节)
├── regression_target.py                               # 回归目标构建 (第3节)
├── wavelet_transform.py                               # 小波变换模块 (第4节)
├── run_complete_analysis.py                           # 完整运行脚本
├── requirements.txt                                   # 依赖包列表
├── task.md                                           # 任务详细规范
└── README.md                                         # 项目说明
```

## 技术实现架构

### 第1节：基础数据注入与准备
- **多模态数据加载**: 使用`scipy.io.loadmat`加载MATLAB格式数据
- **数据结构化**: 采用pandas DataFrame和xarray Dataset组织异构数据
- **声波预处理**: 4阶Butterworth高通滤波(1000Hz截止)，零相位filtfilt实现

### 第2节：高精度时空-方位数据对齐  
- **井眼轨迹重建**: 最小曲率法计算三维井眼坐标(TVD, 北向, 东向)
- **统一深度基准**: 在2732-4132ft研究区间内以0.14ft步长建立统一深度轴
- **方位校正**: 动态计算8个声波接收器的绝对方位角，执行±22.5°扇区匹配

### 第3节：构建量化的回归目标
- **窜槽识别**: 基于Zc<2.5阈值从超声数据生成二值胶结图
- **CSI计算**: 窜槽严重性指数 = 扇区内窜槽点数/总点数，值域[0,1]
- **数据持久化**: HDF5+Pickle双格式保存，支持快速重载

### 第4节：连续小波变换时频分解
- **小波设计**: 复数Morlet小波(cmor1.5-1.0)，覆盖1Hz-30kHz频段
- **尺度计算**: 对数间隔100个尺度，实现多分辨率分析
- **尺度图生成**: |CWT系数|转换为二维时频表示，适配CNN输入

## 安装与运行

### 1. 环境准备

确保Python版本 >= 3.8，然后安装依赖：

```bash
pip install -r requirements.txt
```

### 2. 数据准备

确保以下数据文件位于正确位置：
- `data/raw/CAST.mat` (超声数据，~15MB)
- `data/raw/XSILMR/XSILMR03.mat` (声波数据，~163MB)  
- `data/raw/D2_XSI_RelBearing_Inclination.mat` (方位数据，~106KB)

### 3. 运行分析

执行完整分析流程：

```bash
python run_complete_analysis.py
```

或者运行快速测试：

```bash
python run_complete_analysis.py --test
```

### 4. 逐步运行

如需逐步执行各个模块：

```python
from main_analysis import CementChannelingAnalyzer
from wellpath_alignment import add_alignment_to_analyzer
from regression_target import add_regression_target_to_analyzer
from wavelet_transform import add_wavelet_transform_to_analyzer

# 初始化
analyzer = CementChannelingAnalyzer()
add_alignment_to_analyzer()
add_regression_target_to_analyzer()
add_wavelet_transform_to_analyzer()

# 逐步执行
analyzer.load_data()
analyzer.structure_data()
analyzer.preprocess_sonic_waveforms()
analyzer.run_alignment_section()
analyzer.run_regression_target_section()
analyzer.run_wavelet_transform_section()
```

## 输出文件说明

分析完成后将生成以下文件：

### 可视化图表
- `filtering_effect_comparison.png` - 高通滤波前后波形对比
- `alignment_results.png` - 数据对齐结果四象限图
- `channeling_distribution.png` - 窜槽分布热力图和深度曲线
- `csi_distribution_analysis.png` - CSI分布统计分析四象限图
- `wavelet_scales_design.png` - 小波尺度设计可视化
- `sample_scalograms.png` - 不同CSI等级的样本尺度图对比
- `time_frequency_energy_analysis.png` - 时频能量分布差异分析

### 数据文件
- `processed_data.h5/.pkl` - 处理后的完整数据集
- `scalogram_dataset.npz` - 压缩的尺度图数据集

## 关键参数配置

### 数据处理参数
```python
UNIFIED_DEPTH_RANGE = (2732, 4132)  # ft, 研究深度区间
DEPTH_RESOLUTION = 0.14              # ft, 统一深度步长
CHANNELING_THRESHOLD = 2.5           # 窜槽识别阈值
SECTOR_HALF_WIDTH = 22.5             # 度, 接收器扇区半宽
```

### 滤波参数
```python
CUTOFF_FREQUENCY = 1000              # Hz, 高通滤波截止频率
FILTER_ORDER = 4                     # 滤波器阶数
SAMPLING_RATE = 100000               # Hz, 采样率
```

### 小波变换参数
```python
WAVELET_NAME = 'cmor1.5-1.0'         # 复数Morlet小波
TARGET_FREQ_RANGE = (1, 30000)       # Hz, 目标频率范围
N_SCALES = 100                       # 尺度数量
```

## 理论基础

### 连续小波变换(CWT)
采用复数Morlet小波进行多分辨率时频分析：
```
Wψ(a,b) = ∫ f(t)ψ*((t-b)/a)dt / √a
```
其中ψ为母小波，a为尺度参数，b为时间平移参数。

### 窜槽严重性指数(CSI)
定义为扇区内窜槽点比例：
```
CSI = Σ(Zc < threshold) / N_total
```
提供[0,1]连续值，支持回归建模。

### 最小曲率法井眼轨迹重建
基于测斜数据计算三维井眼坐标：
```
Δx = (ΔMD/2) * (sinI₁cosA₁ + sinI₂cosA₂) * RF
Δy = (ΔMD/2) * (sinI₁sinA₁ + sinI₂sinA₂) * RF  
ΔZ = (ΔMD/2) * (cosI₁ + cosI₂) * RF
```

## 数据质量统计

本项目处理的数据规模：

| 数据类型 | 原始维度 | 处理后维度 | 深度范围 | 备注 |
|---------|---------|-----------|---------|------|
| 超声数据 | 180×24750 | 180×N_unified | 全井段 | 每2°一个方位 |
| 声波数据 | 1024×7108×8 | 1024×N_unified×8 | 全井段 | 8个接收器 |
| 方位数据 | 1×13508 | 1×N_unified | 全井段 | 井斜角+相对方位 |

其中N_unified约为10000个深度点(基于0.14ft步长)。

## 下一步开发计划

1. **第5节**: 基于CNN的窜槽预测回归模型
   - 定制化CNN架构设计
   - 模型训练与验证
   - 性能评估与优化

2. **第6节**: 回归适应性Grad-CAM可解释性分析
   - 梯度计算与权重映射
   - 双视证据图生成
   - 显著性热力图可视化

3. **第7节**: 信号重构与可逆性验证
   - 逆连续小波变换实现
   - 基于掩码的信号重构
   - 最终诊断可视化

## 技术支持

如遇到问题，请检查：

1. **依赖环境**: 确保所有依赖包版本兼容
2. **数据完整性**: 验证.mat文件完整无损
3. **内存需求**: 大规模CWT变换需要足够内存(建议8GB+)
4. **Python版本**: 推荐Python 3.8+

## 引用

如果使用本项目，请引用：

```
基于小波变换与梯度加权可解释性的声波测井水泥窜槽特征识别深度学习框架
AI 助手实现, 2024
```

## 许可证

本项目采用MIT许可证，详见LICENSE文件。 