#!/usr/bin/env python3
"""
增强版数据分析脚本：分析日志文件中的请求到达率和CV值

该脚本分析 /mnt/shared/data/replay-logs-origin.log 文件，计算：
1. 不同时间粒度下的请求到达率
2. 到达间隔时间的变异系数(CV值)
3. 时间分布统计信息
4. 高级流量模式分析
5. 可视化图表和报告
"""

import json
import re
import time
import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import hashlib
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich.panel import Panel
from rich.layout import Layout
import warnings

# 忽略pandas警告
warnings.filterwarnings('ignore', category=FutureWarning)

# Set font for better display
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局变量
console = Console()

class LogAnalyzer:
    """Enhanced log analyzer class"""
    
    def __init__(self, log_file_path: str, sample_start: float = 0.0, sample_end: float = 1.0):
        self.log_file_path = log_file_path
        self.sample_start = sample_start
        self.sample_end = sample_end
        self.timestamps = []
        self.conversation_ids = []
        self.raw_data = []
        self.analysis_cache = {}
        
    def parse_timestamp(self, timestamp_str: str) -> int:
        """解析ISO时间戳为Unix时间戳（纳秒）"""
        try:
            dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
            return int(dt.timestamp() * 1000000000)
        except ValueError as e:
            logger.error(f"解析时间戳错误 {timestamp_str}: {e}")
            return 0
    
    def find_json_objects(self, text: str) -> List[str]:
        """使用栈方法查找文本中的所有JSON对象"""
        objects = []
        stack = []
        start = -1
        
        for i, char in enumerate(text):
            if char == '{':
                if not stack:
                    start = i
                stack.append(char)
            elif char == '}':
                if stack:
                    stack.pop()
                    if not stack:  # 找到完整对象
                        objects.append(text[start:i+1])
        
        return objects
    
    def extract_json_from_log(self, line: str) -> Optional[dict]:
        """从日志行中提取JSON消息"""
        try:
            # 查找行中的所有JSON对象
            json_objects = self.find_json_objects(line)
            if not json_objects:
                return None
                
            # 从后往前尝试解析每个JSON对象
            for json_str in reversed(json_objects):
                try:
                    message_json = json.loads(json_str)
                    if 'message' in message_json:
                        message = message_json['message']
                        if message.startswith('[Log chat request] '):
                            request_str = message[len('[Log chat request] '):]
                            request_json = json.loads(request_str)
                            return request_json
                except json.JSONDecodeError:
                    continue
                    
            return None
                
        except Exception as e:
            logger.error(f"意外错误: {e}")
            return None
    
    def should_process_conversation(self, conversation_id: str) -> bool:
        """基于conversationId确定是否处理该请求"""
        # 如果采样范围覆盖全部，直接返回True
        if self.sample_start == 0.0 and self.sample_end == 1.0:
            return True
        
        # 计算hash值，使其分布在0-1之间
        hash_obj = hashlib.md5(conversation_id.encode())
        hash_bytes = hash_obj.digest()
        # 取前4个字节转换为整数，然后归一化到0-1范围
        hash_int = int.from_bytes(hash_bytes[:4], byteorder='big')
        normalized_hash = hash_int / (2**32)
        
        # 如果哈希值在 [start, end) 区间内，就处理它
        return self.sample_start <= normalized_hash < self.sample_end
    
    def parse_log_file(self) -> None:
        """解析日志文件并提取时间戳和会话ID"""
        logger.info(f"开始解析日志文件: {self.log_file_path}")
        logger.info(f"采样范围: [{self.sample_start*100:.1f}%, {self.sample_end*100:.1f}%)")
        
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        try:
            with open(self.log_file_path, 'r') as fin:
                for line_number, line in enumerate(track(fin, description="解析日志文件")):
                    try:
                        # 提取时间戳
                        timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)', line)
                        if not timestamp_match:
                            continue
                        
                        timestamp = self.parse_timestamp(timestamp_match.group(1))
                        if timestamp == 0:
                            continue

                        # 提取和解析JSON
                        request_data = self.extract_json_from_log(line)
                        if not request_data:
                            continue

                        # 获取会话ID
                        conversation_id = request_data.get('conversationId', '')
                        if not conversation_id:
                            continue
                        
                        # 根据采样率检查是否需要处理该conversationId
                        if not self.should_process_conversation(conversation_id):
                            skipped_count += 1
                            continue

                        # 保存数据
                        self.timestamps.append(timestamp)
                        self.conversation_ids.append(conversation_id)
                        self.raw_data.append({
                            'timestamp': timestamp,
                            'conversation_id': conversation_id,
                            'datetime': datetime.fromtimestamp(timestamp / 1000000000),
                            'line_number': line_number + 1
                        })
                        
                        processed_count += 1
                        
                        if line_number % 10000 == 0 and line_number > 0:
                            logger.info(f"已处理 {line_number} 行，成功解析 {processed_count} 个请求，跳过 {skipped_count} 个，错误 {error_count} 个")
                            
                    except Exception as e:
                        error_count += 1
                        logger.error(f"处理第 {line_number} 行时出错: {e}")
                        continue
                        
            logger.info(f"日志文件解析完成！")
            logger.info(f"总处理行数: {line_number + 1}")
            logger.info(f"成功解析请求数: {processed_count}")
            logger.info(f"跳过请求数: {skipped_count}")
            logger.info(f"错误行数: {error_count}")
            
        except Exception as e:
            logger.error(f"解析日志文件时出错: {e}")
            raise
    
    def calculate_arrival_rate(self, time_granularity: str = '1min') -> pd.DataFrame:
        """计算不同时间粒度下的请求到达率"""
        if not self.timestamps:
            logger.warning("没有时间戳数据，无法计算到达率")
            return pd.DataFrame()
        
        # Check cache
        cache_key = f"arrival_rate_{time_granularity}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        # Create DataFrame
        df = pd.DataFrame(self.raw_data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Set time index
        df = df.set_index('datetime')
        
        # Resample by time granularity and calculate arrival rate
        if time_granularity == '1min':
            resampled = df.resample('1min').size()
        elif time_granularity == '5min':
            resampled = df.resample('5min').size()
        elif time_granularity == '10min':
            resampled = df.resample('10min').size()
        elif time_granularity == '1hour':
            resampled = df.resample('1h').size()
        else:
            resampled = df.resample('1min').size()  # 默认1分钟
        
        # Calculate QPS (requests per second)
        if time_granularity == '1min':
            qps = resampled / 60
        elif time_granularity == '5min':
            qps = resampled / 300
        elif time_granularity == '10min':
            qps = resampled / 600
        elif time_granularity == '1hour':
            qps = resampled / 3600
        else:
            qps = resampled / 60
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'timestamp': resampled.index,
            'request_count': resampled.values,
            'qps': qps.values
        })
        
        # 填充缺失值 - 使用新的pandas语法避免警告
        result_df = result_df.fillna({'request_count': 0, 'qps': 0})
        
        # Cache results
        self.analysis_cache[cache_key] = result_df
        
        return result_df
    
    def calculate_interarrival_times(self) -> pd.Series:
        """计算请求到达间隔时间"""
        if len(self.timestamps) < 2:
            logger.warning("时间戳数量不足，无法计算到达间隔时间")
            return pd.Series()
        
        # Check cache
        if 'interarrival_times' in self.analysis_cache:
            return self.analysis_cache['interarrival_times']
        
        # Sort timestamps
        sorted_timestamps = sorted(self.timestamps)
        
        # Calculate interarrival times (nanoseconds)
        interarrival_ns = np.diff(sorted_timestamps)
        
        # Convert to seconds
        interarrival_seconds = interarrival_ns / 1000000000
        
        # Cache results
        self.analysis_cache['interarrival_times'] = pd.Series(interarrival_seconds)
        
        return self.analysis_cache['interarrival_times']
    
    def calculate_cv(self, data: pd.Series) -> float:
        """计算变异系数 (CV = 标准差/均值)"""
        if len(data) == 0:
            return 0.0
        
        mean_val = data.mean()
        if mean_val == 0:
            return 0.0
        
        std_val = data.std()
        cv = std_val / mean_val
        
        return cv
    
    def analyze_traffic_patterns(self) -> Dict:
        """分析流量模式"""
        if not self.timestamps:
            return {}
        
        # Check cache
        if 'traffic_patterns' in self.analysis_cache:
            return self.analysis_cache['traffic_patterns']
        
        # 计算到达间隔时间
        interarrival_times = self.calculate_interarrival_times()
        
        if len(interarrival_times) == 0:
            return {}
        
        # Basic statistics
        stats = {
            'total_requests': len(self.timestamps),
            'time_span_seconds': (max(self.timestamps) - min(self.timestamps)) / 1000000000,
            'time_span_hours': (max(self.timestamps) - min(self.timestamps)) / 1000000000 / 3600,
            'overall_qps': len(self.timestamps) / ((max(self.timestamps) - min(self.timestamps)) / 1000000000),
            'interarrival_stats': {
                'count': len(interarrival_times),
                'mean': interarrival_times.mean(),
                'std': interarrival_times.std(),
                'min': interarrival_times.min(),
                'max': interarrival_times.max(),
                'median': interarrival_times.median(),
                'p95': interarrival_times.quantile(0.95),
                'p99': interarrival_times.quantile(0.99),
                'cv': self.calculate_cv(interarrival_times)
            }
        }
        
        # Cache results
        self.analysis_cache['traffic_patterns'] = stats
        
        return stats
    
    def detect_anomalies(self) -> Dict:
        """检测流量异常"""
        if not self.timestamps:
            return {}
        
        # 计算不同时间粒度的QPS
        qps_1min = self.calculate_arrival_rate('1min')
        
        if qps_1min.empty:
            return {}
        
        # Use 3-sigma rule to detect anomalies
        qps_values = qps_1min['qps'].values
        mean_qps = np.mean(qps_values)
        std_qps = np.std(qps_values)
        
        # 异常阈值
        upper_threshold = mean_qps + 3 * std_qps
        lower_threshold = max(0, mean_qps - 3 * std_qps)
        
        # 检测异常点
        anomalies = qps_1min[
            (qps_1min['qps'] > upper_threshold) | 
            (qps_1min['qps'] < lower_threshold)
        ]
        
        return {
            'mean_qps': mean_qps,
            'std_qps': std_qps,
            'upper_threshold': upper_threshold,
            'lower_threshold': lower_threshold,
            'anomaly_count': len(anomalies),
            'anomalies': anomalies.to_dict('records') if not anomalies.empty else []
        }
    
    def generate_visualizations(self, output_dir: str = "analysis_results") -> None:
        """Generate enhanced visualization charts"""
        import os
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # Set chart style
        plt.style.use('seaborn-v0_8')
        
        # Create subplot layout
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Enhanced Log File Request Arrival Rate Analysis', fontsize=18, fontweight='bold')
        
        # 1. Time series chart - 1 minute granularity
        ax1 = fig.add_subplot(gs[0, :2])
        df_1min = self.calculate_arrival_rate('1min')
        if not df_1min.empty:
            ax1.plot(df_1min['timestamp'], df_1min['qps'], linewidth=1, alpha=0.8, color='blue')
            ax1.set_title('Request Arrival Rate Time Series (1-min granularity)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('QPS (requests/sec)')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Add anomaly detection
            anomalies = self.detect_anomalies()
            if anomalies and anomalies['anomaly_count'] > 0:
                anomaly_df = pd.DataFrame(anomalies['anomalies'])
                if not anomaly_df.empty:
                    ax1.scatter(anomaly_df['timestamp'], anomaly_df['qps'], 
                               color='red', s=50, alpha=0.7, label=f'Anomalies ({anomalies["anomaly_count"]})')
                    ax1.legend()
        
        # 2. Interarrival time distribution histogram
        ax2 = fig.add_subplot(gs[0, 2])
        interarrival_times = self.calculate_interarrival_times()
        if len(interarrival_times) > 0:
            # Filter outliers (values exceeding 99th percentile)
            p99 = interarrival_times.quantile(0.99)
            filtered_times = interarrival_times[interarrival_times <= p99]
            
            ax2.hist(filtered_times, bins=50, alpha=0.7, edgecolor='black', color='green')
            ax2.set_title('Request Interarrival Time Distribution', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Interarrival Time (seconds)')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
            
            # Add mean line
            mean_time = filtered_times.mean()
            ax2.axvline(mean_time, color='red', linestyle='--', 
                        label=f'Mean: {mean_time:.3f}s')
            ax2.legend()
        
        # 3. QPS comparison at different time granularities
        ax3 = fig.add_subplot(gs[1, :])
        time_granularities = ['1min', '5min', '10min', '1hour']
        qps_data = {}
        
        for granularity in time_granularities:
            df_temp = self.calculate_arrival_rate(granularity)
            if not df_temp.empty:
                qps_data[granularity] = df_temp['qps'].values
        
        if qps_data:
            # Create boxplot
            qps_values = list(qps_data.values())
            qps_labels = list(qps_data.keys())
            bp = ax3.boxplot(qps_values, tick_labels=qps_labels, patch_artist=True)
            
            # Set colors
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax3.set_title('QPS Distribution at Different Time Granularities', fontsize=14, fontweight='bold')
            ax3.set_ylabel('QPS (requests/sec)')
            ax3.grid(True, alpha=0.3)
        
        # 4. 24-hour request distribution
        ax4 = fig.add_subplot(gs[2, :2])
        if self.raw_data:
            df_temp = pd.DataFrame(self.raw_data)
            df_temp['hour'] = df_temp['datetime'].dt.hour
            hourly_counts = df_temp['hour'].value_counts().sort_index()
            
            bars = ax4.bar(hourly_counts.index, hourly_counts.values, alpha=0.7, color='skyblue')
            ax4.set_title('24-Hour Request Distribution', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Hour')
            ax4.set_ylabel('Request Count')
            ax4.set_xticks(range(0, 24))
            ax4.grid(True, alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}', ha='center', va='bottom')
        
        # 5. CV value time trend
        ax5 = fig.add_subplot(gs[2, 2])
        if len(interarrival_times) > 100:
            # Calculate sliding window CV values
            window_size = min(100, len(interarrival_times) // 10)
            cv_values = []
            time_points = []
            
            for i in range(0, len(interarrival_times) - window_size, window_size // 2):
                window_data = interarrival_times.iloc[i:i+window_size]
                cv = self.calculate_cv(window_data)
                cv_values.append(cv)
                time_points.append(i)
            
            if cv_values:
                ax5.plot(time_points, cv_values, marker='o', linewidth=2, color='purple')
                ax5.set_title('CV Value Time Trend', fontsize=14, fontweight='bold')
                ax5.set_xlabel('Time Window')
                ax5.set_ylabel('CV Value')
                ax5.grid(True, alpha=0.3)
                
                # Add CV threshold lines
                ax5.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='Low Variation Threshold')
                ax5.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='Medium Variation Threshold')
                ax5.legend()
        
        # 6. Traffic pattern heatmap
        ax6 = fig.add_subplot(gs[3, :])
        if self.raw_data:
            df_temp = pd.DataFrame(self.raw_data)
            df_temp['hour'] = df_temp['datetime'].dt.hour
            df_temp['minute'] = df_temp['datetime'].dt.minute
            df_temp['minute_group'] = (df_temp['minute'] // 10) * 10  # 10-minute grouping
            
            # Create pivot table
            pivot_table = df_temp.groupby(['hour', 'minute_group']).size().unstack(fill_value=0)
            
            # Draw heatmap
            im = ax6.imshow(pivot_table.values, cmap='YlOrRd', aspect='auto', interpolation='nearest')
            ax6.set_title('24-Hour Traffic Heatmap (10-min granularity)', fontsize=14, fontweight='bold')
            ax6.set_xlabel('Minutes (10-min groups)')
            ax6.set_ylabel('Hour')
            ax6.set_yticks(range(24))
            ax6.set_yticklabels(range(24))
            ax6.set_xticks(range(len(pivot_table.columns)))
            ax6.set_xticklabels([f'{col:02d}' for col in pivot_table.columns])
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax6)
            cbar.set_label('Request Count')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/enhanced_arrival_rate_analysis.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/enhanced_arrival_rate_analysis.pdf", bbox_inches='tight')
        logger.info(f"Enhanced visualization charts saved to: {output_dir}/")
        
        # Close chart to free memory
        plt.close()
    
    def print_analysis_results(self) -> None:
        """Print enhanced analysis results"""
        if not self.timestamps:
            console.print("[red]No data to analyze[/red]")
            return
        
        # Analyze traffic patterns
        stats = self.analyze_traffic_patterns()
        
        if not stats:
            console.print("[red]Unable to analyze traffic patterns[/red]")
            return
        
        # Detect anomalies
        anomalies = self.detect_anomalies()
        
        # Create result tables
        console.print("\n[bold blue]=== Enhanced Log File Analysis Results ===[/bold blue]")
        
        # Basic information table
        basic_table = Table(title="Basic Information", show_header=True, header_style="bold magenta")
        basic_table.add_column("Metric", style="cyan")
        basic_table.add_column("Value", style="green")
        
        basic_table.add_row("Total Requests", f"{stats['total_requests']:,}")
        basic_table.add_row("Time Span", f"{stats['time_span_hours']:.2f} hours")
        basic_table.add_row("Overall QPS", f"{stats['overall_qps']:.2f}")
        
        if anomalies:
            basic_table.add_row("Anomaly Detection", f"{anomalies['anomaly_count']} anomalies")
        
        console.print(basic_table)
        
        # Interarrival time statistics table
        interarrival_stats = stats['interarrival_stats']
        interarrival_table = Table(title="Interarrival Time Statistics", show_header=True, header_style="bold magenta")
        interarrival_table.add_column("Metric", style="cyan")
        interarrival_table.add_column("Value", style="green")
        interarrival_table.add_column("Unit", style="yellow")
        
        interarrival_table.add_row("Sample Count", f"{interarrival_stats['count']:,}", "")
        interarrival_table.add_row("Mean", f"{interarrival_stats['mean']:.6f}", "seconds")
        interarrival_table.add_row("Standard Deviation", f"{interarrival_stats['std']:.6f}", "seconds")
        interarrival_table.add_row("Minimum", f"{interarrival_stats['min']:.6f}", "seconds")
        interarrival_table.add_row("Maximum", f"{interarrival_stats['max']:.6f}", "seconds")
        interarrival_table.add_row("Median", f"{interarrival_stats['median']:.6f}", "seconds")
        interarrival_table.add_row("P95", f"{interarrival_stats['p95']:.6f}", "seconds")
        interarrival_table.add_row("P99", f"{interarrival_stats['p99']:.6f}", "seconds")
        interarrival_table.add_row("Coefficient of Variation (CV)", f"{interarrival_stats['cv']:.4f}", "")
        
        console.print(interarrival_table)
        
        # CV value interpretation
        cv = interarrival_stats['cv']
        if cv < 0.5:
            cv_level = "Low Variation"
            cv_color = "green"
        elif cv < 1.0:
            cv_level = "Medium Variation"
            cv_color = "yellow"
        else:
            cv_level = "High Variation"
            cv_color = "red"
        
        console.print(f"\n[bold]Coefficient of Variation (CV) Analysis:[/bold]")
        console.print(f"CV Value: [{cv_color}]{cv:.4f}[/{cv_color}] - {cv_level}")
        console.print(f"CV = Standard Deviation / Mean = {interarrival_stats['std']:.6f} / {interarrival_stats['mean']:.6f}")
        
        if cv < 1.0:
            console.print("Explanation: Request arrival times are relatively regular, traffic is stable")
        else:
            console.print("Explanation: Request arrival times vary significantly, traffic is volatile")
        
        # Anomaly detection results
        if anomalies and anomalies['anomaly_count'] > 0:
            console.print(f"\n[bold red]Anomaly Detection Results:[/bold red]")
            console.print(f"Detected {anomalies['anomaly_count']} anomalies")
            console.print(f"Anomaly threshold: [{anomalies['lower_threshold']:.2f}, {anomalies['upper_threshold']:.2f}] QPS")
            
            # Display first few anomalies
            if anomalies['anomalies']:
                anomaly_table = Table(title="Anomaly Details", show_header=True, header_style="bold red")
                anomaly_table.add_column("Time", style="cyan")
                anomaly_table.add_column("QPS", style="red")
                anomaly_table.add_column("Request Count", style="yellow")
                
                for i, anomaly in enumerate(anomalies['anomalies'][:5]):  # Only show first 5
                    anomaly_table.add_row(
                        str(anomaly['timestamp']),
                        f"{anomaly['qps']:.2f}",
                        str(anomaly['request_count'])
                    )
                
                if len(anomalies['anomalies']) > 5:
                    anomaly_table.add_row("...", "...", "...")
                
                console.print(anomaly_table)
        
        # QPS statistics at different time granularities
        console.print("\n[bold]QPS Statistics at Different Time Granularities:[/bold]")
        granularities = ['1min', '5min', '10min', '1hour']
        
        qps_table = Table(title="QPS Statistics", show_header=True, header_style="bold magenta")
        qps_table.add_column("Time Granularity", style="cyan")
        qps_table.add_column("Average QPS", style="green")
        qps_table.add_column("Maximum QPS", style="green")
        qps_table.add_column("Minimum QPS", style="green")
        qps_table.add_column("Standard Deviation", style="green")
        
        for granularity in granularities:
            df_temp = self.calculate_arrival_rate(granularity)
            if not df_temp.empty:
                qps_values = df_temp['qps'].values
                qps_table.add_row(
                    granularity,
                    f"{qps_values.mean():.2f}",
                    f"{qps_values.max():.2f}",
                    f"{qps_values.min():.2f}",
                    f"{qps_values.std():.2f}"
                )
        
        console.print(qps_table)
    
    def export_results(self, output_dir: str = "analysis_results") -> None:
        """Export enhanced analysis results to files"""
        import os
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # Export interarrival time data
        interarrival_times = self.calculate_interarrival_times()
        if len(interarrival_times) > 0:
            interarrival_df = pd.DataFrame({
                'interarrival_time_seconds': interarrival_times.values
            })
            interarrival_file = f"{output_dir}/enhanced_interarrival_times.csv"
            interarrival_df.to_csv(interarrival_file, index=False)
            logger.info(f"Enhanced interarrival time data exported to: {interarrival_file}")
        
        # Export QPS data at different time granularities
        for granularity in ['1min', '5min', '10min', '1hour']:
            df_temp = self.calculate_arrival_rate(granularity)
            if not df_temp.empty:
                qps_file = f"{output_dir}/enhanced_qps_{granularity}.csv"
                df_temp.to_csv(qps_file, index=False)
                logger.info(f"Enhanced {granularity} granularity QPS data exported to: {qps_file}")
        
        # Export anomaly detection results
        anomalies = self.detect_anomalies()
        if anomalies and anomalies['anomaly_count'] > 0:
            anomaly_file = f"{output_dir}/anomaly_detection.csv"
            anomaly_df = pd.DataFrame(anomalies['anomalies'])
            anomaly_df.to_csv(anomaly_file, index=False)
            logger.info(f"Anomaly detection results exported to: {anomaly_file}")
        
        # Export statistics summary
        stats = self.analyze_traffic_patterns()
        if stats:
            stats_file = f"{output_dir}/enhanced_analysis_summary.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                # Convert numpy and pandas types to Python native types for JSON serialization
                def convert_types(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif hasattr(obj, 'isoformat'):  # Handle pandas Timestamp
                        return obj.isoformat()
                    elif hasattr(obj, 'timestamp'):  # Handle datetime objects
                        return obj.timestamp()
                    return obj
                
                # Recursive conversion
                def recursive_convert(obj):
                    if isinstance(obj, dict):
                        return {k: recursive_convert(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [recursive_convert(v) for v in obj]
                    else:
                        return convert_types(obj)
                
                # Merge statistics and anomaly detection results
                enhanced_stats = {
                    'traffic_patterns': recursive_convert(stats),
                    'anomaly_detection': recursive_convert(anomalies) if anomalies else None
                }
                
                json.dump(enhanced_stats, f, indent=2, ensure_ascii=False)
                logger.info(f"Enhanced analysis summary exported to: {stats_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Enhanced log file request arrival rate and CV value analysis')
    parser.add_argument('--input', '-i', default="/mnt/shared/data/replay-logs-origin.log",
                        help='Input log file path (default: /mnt/shared/data/replay-logs-origin.log)')
    parser.add_argument('--sample-range', type=float, nargs=2, default=[0.0, 1.0],
                        metavar=('START', 'END'),
                        help='Sample range [START, END) to control request percentage (e.g., 0.0 0.2). Default: [0.0, 1.0]')
    parser.add_argument('--output-dir', '-o', default='analysis_results',
                        help='Output directory (default: analysis_results)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Do not generate visualization charts')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable detailed logging')
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure sample range is valid
    sample_start, sample_end = args.sample_range
    if not (0.0 <= sample_start < sample_end <= 1.0):
        raise ValueError(f"Invalid sample range [{sample_start}, {sample_end}). Must be 0.0 <= START < END <= 1.0")
    
    try:
        # Create analyzer
        analyzer = LogAnalyzer(args.input, sample_start, sample_end)
        console.print("[bold green]Using enhanced log analyzer[/bold green]")
        
        # Parse log file
        analyzer.parse_log_file()
        
        # Print analysis results
        analyzer.print_analysis_results()
        
        # Generate visualization charts
        if not args.no_plot:
            analyzer.generate_visualizations(args.output_dir)
        
        # Export results
        analyzer.export_results(args.output_dir)
        
        console.print(f"\n[bold green]Analysis completed! Results saved to: {args.output_dir}/[/bold green]")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main() 