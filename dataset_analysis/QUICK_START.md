# 快速开始指南

## 🚀 5分钟快速上手

### 1. 安装依赖
```bash
cd log_analysis
pip install -r requirements_analysis.txt
```

### 2. 运行分析
```bash
# 推荐：增强版分析（包含异常检测）
./run_enhanced_analysis.sh

# 或者：标准版分析
./run_analysis.sh
```

### 3. 查看结果
- 控制台输出：实时统计信息
- 图表文件：`analysis_results/` 目录
- 数据文件：CSV和JSON格式

## 📊 分析结果解读

### CV值含义
- **CV < 0.5**: 流量稳定，系统负载均衡
- **0.5 ≤ CV < 1.0**: 流量较稳定，偶有波动
- **CV ≥ 1.0**: 流量波动大，可能存在突发流量

### 异常检测
- 自动识别异常的QPS峰值
- 标记流量异常的时间点
- 提供异常阈值和统计信息

## 🔧 常用命令

```bash
# 分析前10%的请求
python3 analyze_arrival_rate_enhanced.py --enhanced --sample-range 0.0 0.1

# 指定输出目录
python3 analyze_arrival_rate_enhanced.py --enhanced -o my_results

# 只输出统计结果，不生成图表
python3 analyze_arrival_rate_enhanced.py --enhanced --no-plot

# 启用详细日志
python3 analyze_arrival_rate_enhanced.py --enhanced -v
```

## 📁 输出文件说明

- `enhanced_arrival_rate_analysis.png/pdf`: 综合分析图表
- `enhanced_qps_*.csv`: 不同时间粒度的QPS数据
- `enhanced_interarrival_times.csv`: 到达间隔时间数据
- `anomaly_detection.csv`: 异常检测结果
- `enhanced_analysis_summary.json`: 完整分析摘要

## ❓ 常见问题

**Q: 内存不足怎么办？**
A: 使用 `--sample-range 0.0 0.1` 减少采样范围

**Q: 图表显示中文乱码？**
A: 脚本已配置多种字体，如果仍有问题请安装SimHei字体

**Q: 处理速度慢？**
A: 增强版脚本包含缓存机制，重复运行会更快

## 📞 获取帮助

查看完整文档：`README.md`
运行测试：`python3 test_analyzer.py` (如果存在)

---

**提示**: 首次使用建议运行增强版分析，功能更全面！ 