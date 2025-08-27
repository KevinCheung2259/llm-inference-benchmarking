# 日志文件请求到达率分析工具包

## 概述

这是一个完整的日志文件请求到达率分析工具包，专门用于分析 `/mnt/shared/data/replay-logs-origin.log` 文件中的请求到达率和变异系数(CV值)。

## 文件结构

```
log_analysis/
├── README.md                           # 本说明文档
├── analyze_arrival_rate.py             # 标准版分析脚本
├── analyze_arrival_rate_enhanced.py    # 增强版分析脚本
├── requirements_analysis.txt            # Python依赖包列表
├── run_analysis.sh                     # 标准版启动脚本
├── run_enhanced_analysis.sh            # 增强版启动脚本
├── analysis_config.json                # 配置文件示例
└── analysis_results/                   # 分析结果输出目录
```

## 功能特性

### 1. 到达率分析
- **多时间粒度**: 1分钟、5分钟、10分钟、1小时
- **QPS计算**: 每个时间窗口的请求每秒数
- **统计分布**: 均值、最大值、最小值、标准差

### 2. CV值计算与分析
- **变异系数**: CV = 标准差/均值
- **分级标准**:
  - CV < 0.5: 低变异，流量稳定
  - 0.5 ≤ CV < 1.0: 中等变异，流量较稳定
  - CV ≥ 1.0: 高变异，流量波动明显

### 3. 流量模式识别
- **时间分布**: 24小时请求分布图
- **间隔分析**: 到达间隔时间直方图
- **趋势分析**: 时间序列趋势图

### 4. 增强版功能
- **异常检测**: 使用3-sigma规则检测流量异常
- **CV值趋势**: 滑动窗口CV值变化分析
- **流量热力图**: 24小时×10分钟粒度的流量分布
- **缓存机制**: 避免重复计算，提升性能

## 安装依赖

```bash
cd log_analysis
pip install -r requirements_analysis.txt
```

## 使用方法

### 快速开始

#### 标准版分析
```bash
# 一键运行
./run_analysis.sh

# 或直接运行
python3 analyze_arrival_rate.py
```

#### 增强版分析
```bash
# 一键运行
./run_enhanced_analysis.sh

# 或直接运行
python3 analyze_arrival_rate_enhanced.py --enhanced
```

### 高级用法

```bash
# 分析前20%的请求
python3 analyze_arrival_rate.py --sample-range 0.0 0.2

# 指定输出目录
python3 analyze_arrival_rate.py -o my_results

# 只输出统计结果，不生成图表
python3 analyze_arrival_rate.py --no-plot

# 启用详细日志
python3 analyze_arrival_rate.py -v
```

### 命令行参数说明

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | `-i` | `/mnt/shared/data/replay-logs-origin.log` | 输入日志文件路径 |
| `--sample-range` | - | `[0.0, 1.0]` | 采样范围 [START, END) |
| `--output-dir` | `-o` | `analysis_results` | 输出目录 |
| `--no-plot` | - | False | 不生成图表，只输出统计结果 |
| `--verbose` | `-v` | False | 启用详细日志 |
| `--enhanced` | - | False | 使用增强版分析功能 |

## 输出结果

### 1. 控制台输出
- 基本统计信息表格
- 到达间隔时间统计表格
- CV值分析和解释
- 不同时间粒度的QPS统计
- 异常检测结果（增强版）

### 2. 文件输出
- `interarrival_times.csv`: 到达间隔时间数据
- `qps_*.csv`: 不同时间粒度的QPS数据
- `analysis_summary.json`: 统计摘要
- `arrival_rate_analysis.png/pdf`: 可视化图表
- `enhanced_*.csv`: 增强版分析数据（增强版）
- `anomaly_detection.csv`: 异常检测结果（增强版）

## 版本对比

### 标准版 vs 增强版

| 功能 | 标准版 | 增强版 |
|------|--------|--------|
| 基本分析 | ✅ | ✅ |
| 可视化图表 | 4个子图 | 6个子图 |
| 异常检测 | ❌ | ✅ |
| CV值趋势 | ❌ | ✅ |
| 流量热力图 | ❌ | ✅ |
| 缓存机制 | ❌ | ✅ |
| 性能优化 | 基础 | 高级 |

## 技术架构

### 1. 日志解析
- **时间戳解析**: ISO 8601格式支持
- **JSON提取**: 栈方法解析嵌套结构
- **采样机制**: MD5哈希确保随机性

### 2. 数据分析
- **流式处理**: 内存占用可控
- **pandas集成**: 高效的数据分析
- **统计计算**: 完整的描述性统计

### 3. 可视化
- **matplotlib**: 高质量图表生成
- **中文字体**: 支持中文显示
- **多格式输出**: PNG和PDF

## 性能特点

### 1. 内存优化
- 流式处理大文件
- 采样机制控制内存使用
- 及时释放临时数据

### 2. 处理速度
- 高效的JSON解析
- pandas向量化计算
- 缓存机制（增强版）

### 3. 可扩展性
- 模块化设计
- 配置驱动
- 易于添加新功能

## 适用场景

### 1. 性能分析
- 系统负载评估
- 流量模式识别
- 容量规划

### 2. 问题诊断
- 突发流量检测
- 负载不均衡分析
- 性能瓶颈识别

### 3. 监控告警
- CV值阈值设置
- QPS异常检测
- 趋势分析

## 故障排除

### 常见问题

#### 1. 字体显示问题
```python
# 脚本已配置多种字体选项
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
```

#### 2. 内存不足
```bash
# 使用采样减少内存使用
python3 analyze_arrival_rate.py --sample-range 0.0 0.1
```

#### 3. 依赖包冲突
```bash
# 创建虚拟环境
python3 -m venv analysis_env
source analysis_env/bin/activate
pip install -r requirements_analysis.txt
```

## 示例输出

### 基本信息
```
=== 日志文件分析结果 ===
总请求数: 1,234,567
时间跨度: 24.50 小时
整体QPS: 13.98
```

### CV值分析
```
变异系数(CV): 0.8234 - 中等变异
说明：请求到达时间相对规律，流量较为稳定
```

### 异常检测（增强版）
```
异常检测结果:
检测到 15 个异常点
异常阈值: [0.00, 45.67] QPS
```

## 下一步建议

### 1. 立即使用
```bash
cd log_analysis
./run_enhanced_analysis.sh  # 推荐使用增强版
```

### 2. 自定义分析
- 调整采样范围
- 修改时间粒度
- 设置CV阈值

### 3. 集成扩展
- 添加到CI/CD流程
- 集成监控系统
- 自动化报告生成

## 总结

这个工具包提供了完整的日志文件请求到达率分析能力：

✅ **完整的分析功能** - 到达率、CV值、流量模式  
✅ **高质量可视化** - 多种图表和格式  
✅ **灵活的配置** - 采样、粒度、输出选项  
✅ **异常检测** - 智能识别流量异常（增强版）  
✅ **性能优化** - 缓存机制和内存优化  
✅ **易于使用** - 一键启动、交互式运行  

现在您可以开始分析日志文件，深入了解系统的流量特征和性能表现！ 