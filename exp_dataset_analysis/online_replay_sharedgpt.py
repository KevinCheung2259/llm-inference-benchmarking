import json
import re
import time
import threading
import queue
import numpy as np
import requests
import asyncio
import aiohttp
from itertools import groupby
from datetime import datetime
import logging
from typing import Optional, Dict, Any, List
import hashlib
import argparse
import openai
from num2words import num2words
import os
import pandas as pd
from rich.console import Console
from rich.table import Table
import math
import sys
import hashlib
from collections import Counter
import uuid
import signal
from datasets import load_dataset

# 全局变量
global_result_collector = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加对 openai 和 httpx 日志的控制
openai_logger = logging.getLogger("openai")
httpx_logger = logging.getLogger("httpx")

def set_logging_level(verbose):
    """设置日志级别"""
    if not verbose:
        logger.setLevel(logging.WARNING)
        openai_logger.setLevel(logging.WARNING)  # 控制 openai 的日志
        httpx_logger.setLevel(logging.WARNING)   # 控制 HTTP 请求的日志
    else:
        logger.setLevel(logging.INFO)
        openai_logger.setLevel(logging.INFO)
        httpx_logger.setLevel(logging.INFO)

# Job queue for storing parsed requests
job_queue = queue.PriorityQueue()

class ReplayJob:
    """Class representing a job to be replayed."""
    def __init__(self, timestamp: int, url: str, headers: Dict[str, str], body: Dict[str, Any], conversation_id: str, use_chat: bool = True):
        self.timestamp = timestamp
        self.url = url
        self.headers = headers
        self.body = body
        self.use_chat = use_chat
        
        # Round timestamp to seconds for grouping
        self.second_timestamp = timestamp // 1000000000
        
        # Get conversation ID for sampling
        self.conversation_id = conversation_id
        
        # 添加唯一会话ID用于追踪每个请求
        self.request_id = str(uuid.uuid4())
    
    def __lt__(self, other):
        return self.timestamp < other.timestamp

def parse_timestamp(timestamp_str: str) -> int:
    """Convert ISO timestamp to Unix timestamp in nanoseconds."""
    try:
        dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
        return int(dt.timestamp() * 1000000000)  # Convert to nanoseconds
    except ValueError as e:
        logger.error(f"Error parsing timestamp {timestamp_str}: {e}")
        return 0

def find_json_objects(text: str) -> list:
    """Find all JSON objects in text using a stack-based approach."""
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
                if not stack:  # Complete object found
                    objects.append(text[start:i+1])
    
    return objects

def extract_json_from_log(line: str) -> Optional[dict]:
    """Extract JSON message from log line."""
    try:
        # Find all JSON objects in the line
        json_objects = find_json_objects(line)
        if not json_objects:
            logger.debug("No JSON objects found in line")
            return None
            
        # Try to parse each JSON object from last to first
        for json_str in reversed(json_objects):
            try:
                message_json = json.loads(json_str)
                if 'message' in message_json:
                    message = message_json['message']
                    if message.startswith('[Log chat request] '):
                        request_str = message[len('[Log chat request] '):]
                        request_json = json.loads(request_str)
                        logger.debug("Successfully parsed JSON")
                        return request_json
            except json.JSONDecodeError:
                continue
                
        logger.debug("No valid message JSON found")
        return None
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None

def should_process_conversation(conversation_id: str, sample_start: float, sample_end: float) -> bool:
    """基于conversationId确定是否处理该请求，根据采样范围[start, end)进行筛选。"""
    # Ensure range is valid
    if sample_start >= sample_end or sample_start < 0.0 or sample_end > 1.0:
         logger.warning(f"Invalid sample range [{sample_start}, {sample_end}). Skipping filtering.")
         return True # Or handle as an error / default behavior

    # If range covers everything
    if sample_start == 0.0 and sample_end == 1.0:
        return True
    
    # 计算hash值，使其分布在0-1之间
    hash_obj = hashlib.md5(conversation_id.encode())
    hash_bytes = hash_obj.digest()
    # 取前4个字节转换为整数，然后归一化到0-1范围
    hash_int = int.from_bytes(hash_bytes[:4], byteorder='big')
    normalized_hash = hash_int / (2**32)
    
    # 如果哈希值在 [start, end) 区间内，就处理它
    return sample_start <= normalized_hash < sample_end

def process_log_line(line: str, sample_start: float = 0.0,
                      sample_end: float = 1.0, ep_config: dict = None) -> Optional[ReplayJob]:
    """Process a single log line and convert it to a ReplayJob."""
    try:
        # Extract timestamp
        timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)', line)
        if not timestamp_match:
            logger.debug("No timestamp match found")
            return None
        
        timestamp = parse_timestamp(timestamp_match.group(1))
        if timestamp == 0:
            return None

        # Extract and parse JSON
        request_data = extract_json_from_log(line)
        # print("request_data", request_data)
        if not request_data:
            logger.debug("No request data extracted")
            return None

        # Prepare headers
        conversation_id = request_data.get('conversationId', '')
        
        # 根据采样率检查是否需要处理该conversationId
        if not should_process_conversation(conversation_id, sample_start, sample_end):
            logger.debug(f"Skipping conversation {conversation_id} due to sampling range [{sample_start}, {sample_end}) - hash: {hashlib.md5(conversation_id.encode()).hexdigest()[:8]}") # Log hash for debugging
            return None

        # 使用传入的配置
        if ep_config is None:
            ep_config = {
                "api_base": "http://localhost:8080/v1",
                "api_key": "default_key",
                "model": "Nitral-AI/Captain-Eris_Violet-V0.420-12B",
                "use_chat": True
            }

        # 构造请求头
        headers = {
            'Authorization': f'Bearer {ep_config["api_key"]}',
            'Content-Type': 'application/json'
        }

        # 构造请求体
        if ep_config.get("use_chat", True):
            messages = request_data['body'].get('prompt', [])
            # 确保messages不为空
            if not messages:
                logger.warning(f"Empty messages for conversation {conversation_id}, skipping")
                return None
            
            body = {
                "model": ep_config["model"],
                "messages": messages,
                "stream": True,
                "max_tokens": ep_config.get("max_tokens", 200)
            }
            url = f"{ep_config['api_base'].rstrip('/')}/chat/completions"
        else:
            # 对于非chat模式，构造一个包含单个消息的messages数组
            body = {
                "model": ep_config["model"],
                "messages": [request_data['body'].get('prompt', '')],
                "stream": True,
                "max_tokens": ep_config.get("max_tokens", 200)
            }
            url = f"{ep_config['api_base'].rstrip('/')}/completions"

        # 创建ReplayJob
        job = ReplayJob(
            timestamp=timestamp,
            url=url,
            headers=headers,
            body=body,
            conversation_id=conversation_id,
            use_chat=ep_config.get("use_chat", False)  # 使用配置中的use_chat参数
        )
        
        return job
        
    except Exception as e:
        logger.error(f"Error processing line: {e}")
        return None

def process_dataset_item(item: dict, sample_start: float = 0.0, sample_end: float = 1.0, 
                         ep_config: dict = None) -> Optional[ReplayJob]:
    """Process a single dataset item and convert it to a ReplayJob."""
    try:
        conversation_id = str(uuid.uuid4())
        if not should_process_conversation(conversation_id, sample_start, sample_end):
            logger.debug(f"Skipping conversation {conversation_id} due to sampling range [{sample_start}, {sample_end}) - hash: {hashlib.md5(conversation_id.encode()).hexdigest()[:8]}")
            return None

        if ep_config is None:
            ep_config = {
                "api_base": "http://localhost:8080/v1",
                "api_key": "default_key",
                "model": "Nitral-AI/Captain-Eris_Violet-V0.420-12B",
                "use_chat": True
            }
        headers = {
            'Authorization': f'Bearer {ep_config["api_key"]}',
            'Content-Type': 'application/json'
        }
        if ep_config.get('dataset') == 'simplescaling/s1K':
            messages = [
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["solution"]}
            ]
            # print("messages:", messages)
        else:
            conversations = item.get('conversations', [])
            messages = [{"role": conv["from"], "content": conv["value"]} for conv in conversations]
        if ep_config.get("use_chat", True):
            if not messages:
                logger.warning(f"Empty messages for conversation {conversation_id}, skipping")
                return None
            body = {
                "model": ep_config["model"],
                "messages": messages,
                "stream": True,
                "max_tokens": ep_config.get("max_tokens", 200)
            }
            url = f"{ep_config['api_base'].rstrip('/')}/chat/completions"
        else:
            prompt = messages[0]["content"] if messages else ""
            body = {
                "model": ep_config["model"],
                "messages": [prompt],
                "stream": True,
                "max_tokens": ep_config.get("max_tokens", 200)
            }
            url = f"{ep_config['api_base'].rstrip('/')}/completions"

        timestamp = int(time.time() * 1e9)

        return ReplayJob(timestamp, url, headers, body, conversation_id, ep_config.get("use_chat", False))
    except Exception as e:
        logger.error(f"Error processing dataset item: {e}")
        return None
    
def log_reader_thread(input_file: str, preload_time: int = 180, sample_start: float = 0.0, 
                      sample_end: float = 1.0, ep_config: dict = None, dataset:str="shibing624/sharegpt_gpt4"):
    """Thread A: Read log file and add jobs to the queue."""
    try:
        logger.info(f"Starting log reader thread with {preload_time} seconds preload time and sample range [{sample_start*100:.1f}%, {sample_end*100:.1f}%)")
        job_count = 0
        skipped_count = 0
        last_timestamp = None

        if True:
            ds = load_dataset(dataset, split="train")
            logger.info("Dataset loaded successfully")
            for item in ds:
                job = process_dataset_item(item, sample_start, sample_end, ep_config)
                if job:
                    job_queue.put(job)
                    job_count += 1
                    logger.debug(f"Added job to queue: {job.conversation_id}")
            logger.info(f"Finished reading log file, added {job_count} jobs")
            return 
        
        with open(input_file, 'r') as fin:
            for line_number, line in enumerate(fin, 1):
                try:
                    logger.debug(f"Processing line {line_number}")
                    job = process_log_line(line.strip(), sample_start, sample_end, ep_config)
                    if job:
                        # Add job to queue
                        job_queue.put(job)
                        job_count += 1
                        last_timestamp = job.timestamp
                        
                        logger.debug(f"Added job for timestamp {job.timestamp}")
                    else:
                        logger.debug(f"No job created for line {line_number}")
                        skipped_count += 1
                except Exception as e:
                    logger.error(f"Error processing line {line_number}: {e}")
                    continue
                
                if line_number % 1000 == 0:
                    logger.info(f"Processed {line_number} lines, added {job_count} jobs, skipped {skipped_count} to queue")
        
        logger.info(f"Finished reading log file, added {job_count} jobs, skipped {skipped_count} to queue")
        
    except Exception as e:
        logger.error(f"Error in log reader thread: {e}")
        raise

# 全局会话对象管理类
class ClientManager:
    def __init__(self):
        self._client = None
        self._background_tasks = set()
        self._lock = threading.Lock()
    
    async def get_client(self, ep_config: dict) -> openai.AsyncOpenAI:
        """获取或创建client实例"""
        with self._lock:
            if self._client is None:
                self._client = openai.AsyncOpenAI(
                    base_url=ep_config["api_base"],
                    api_key=ep_config["api_key"]
                )
            return self._client
    
    def add_background_task(self, task: asyncio.Task):
        """添加后台任务"""
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
    
    async def cleanup(self):
        """清理所有资源"""
        with self._lock:
            if self._client:
                await self._client.close()
                self._client = None
            
            # 等待所有后台任务完成
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
                self._background_tasks.clear()

# 创建全局ClientManager实例
client_manager = ClientManager()

async def send_request(client, job):
    """Send a single request asynchronously and collect metrics."""
    try:
        start_time = time.perf_counter()
        ttft = None
        tokens_in = 0
        tokens_out = 0
        
        # Add headers for tracking
        extra_headers = {
            "X-Flow-Conversation-Id": str(job.conversation_id) if job.conversation_id else "",
            "X-Request-Id": job.request_id  # vllm读取这个字段作为request_id，添加request_id到请求头，用于全链路追踪
        }
        extra_body = {}
        # 采样参数全部放到 extra_body
        extra_body.update({
            "min_p": 0.02,
            "top_p": 1,
            "top_k": -1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "repetition_penalty": 1,
            "temperature": 0.8
        })
        
        if job.use_chat:
            if not job.body.get("messages"):
                logger.warning("Empty messages array, skipping request")
                return (job.request_id, "Exception", -1, -1, -1, -1, "Empty messages")
            # 当模型名包含"MiniCPM4"，显式设置 extra_body={"add_special_tokens": True}
            if "MiniCPM4" in job.body.get("model"):
                extra_body["add_special_tokens"] = True
            
            response = await client.chat.completions.create(
                model=job.body.get("model"),
                messages=[
                    {"role": "user", "content": job.body.get("messages")}
                ],
                max_tokens=job.body.get("max_tokens", 200),
                temperature=0,
                stream=True,
                stream_options={"include_usage": True},
                extra_headers=extra_headers,
                extra_body=extra_body,
            )
        else:
            response = await client.completions.create(
                model=job.body.get("model"),
                prompt=job.body.get("messages"),
                max_tokens=job.body.get("max_tokens", 200),
                temperature=0,
                stream=True,
                stream_options={"include_usage": True},
                extra_headers=extra_headers,
                extra_body=extra_body,
            )
            
        words = ""
        async for tok in response:
            if not tok.choices:
                continue
            if job.use_chat:
                delta = tok.choices[0].delta
                if delta.content:
                    if ttft is None:
                        ttft = time.perf_counter() - start_time
                    words += delta.content
            else:
                delta = tok.choices[0]
                if delta.text:
                    if ttft is None:
                        ttft = time.perf_counter() - start_time
                    words += delta.text
                    
        tokens_in = tok.usage.prompt_tokens
        tokens_out = tok.usage.completion_tokens
        total_time = time.perf_counter() - start_time
        
        return (job.request_id, "OK", ttft, total_time, tokens_in, tokens_out, "")
        
    except asyncio.TimeoutError:
        logger.error("Request timed out after 2s")
        return (job.request_id, "Exception", -1, -1, -1, -1, "Timeout")
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return (job.request_id, "Exception", -1, -1, -1, -1, str(e))
    

class ResultCollector:
    """用于收集和分析请求结果的类"""
    def __init__(self, ep_config, round_duration, max_rounds=None, detailed_logs=False):
        self.results_queue = queue.Queue()
        self.query_results = []
        self.elts = []
        self.jobs_processed = 0
        self.round_start_time = time.perf_counter()
        self.ep_config = ep_config
        self.round_duration = round_duration
        self.total_requests = 0
        self.successful_requests = 0
        self.current_round = 0
        self.max_rounds = max_rounds
        self.detailed_logs = detailed_logs
        
        # 用于详细日志的数据结构
        if detailed_logs:
            self.detailed_results = {}
            
            # 创建CSV文件并写入表头
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 如果detailed_logs是字符串路径，则使用指定路径
            if isinstance(detailed_logs, str) and detailed_logs is not True:
                # 确保目录存在
                log_dir = os.path.dirname(detailed_logs)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir)
            else:
                # 使用默认路径：当前文件所在文件夹下的log文件夹
                log_dir = os.path.join(os.path.dirname(__file__), "logs")
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
            
            self.csv_filename = f"{log_dir}/detailed_results_{timestamp}.csv"
            self.csv_file = open(self.csv_filename, 'w')
            self.csv_file.write("request_id,conversation_id,send_time,ttft_time,total_time,tokens_in,tokens_out,ttft,tpot\n")
            self.csv_file.flush()
            logger.info(f"已创建详细日志文件: {self.csv_filename}")

    def task_done_callback(self, task):
        """处理异步任务完成的回调函数"""
        try:
            result = task.result()
            self.results_queue.put(result)
            self.total_requests += 1
            if result[0] == "OK":
                self.successful_requests += 1
        except Exception as e:
            logger.error(f"Error in callback: {e}")
            self.results_queue.put(("Exception", -1, -1, -1, -1, str(e)))
            self.total_requests += 1

    def get_success_rate(self):
        """计算成功率"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    def collect_results(self):
        """收集队列中的结果"""
        while True:
            try:
                result = self.results_queue.get_nowait()
                self.query_results.append(result)
            except queue.Empty:
                break

    def check_and_report_metrics(self, qps=None, concur_requests=None):
        """检查是否需要报告指标并重置统计"""
        current_time = time.perf_counter()
        if current_time - self.round_start_time >= self.round_duration:
            self.current_round += 1
            elapsed_time = current_time - self.round_start_time
            self.elts.append(elapsed_time)
            actual_qps = self.jobs_processed / elapsed_time
            
            # 分析结果
            results_analysis(
                self.query_results,
                self.elts,
                self.ep_config["model"],
                qps=qps,  # 使用传入的目标QPS
                actual_qps=actual_qps,  # 同时传入实际QPS
                concur_requests=concur_requests,
                json_output=args.json_output,
            )
            
            # 重置统计
            self.query_results = []
            self.jobs_processed = 0
            self.round_start_time = current_time

            # 检查是否达到最大轮数
            if self.max_rounds is not None and self.current_round >= self.max_rounds:
                # 关闭详细日志文件
                if self.detailed_logs and hasattr(self, 'csv_file') and not self.csv_file.closed:
                    self.csv_file.close()
                    logger.info(f"已关闭详细日志文件: {self.csv_filename}")
                
                logger.info(f"达到最大轮数 {self.max_rounds}，直接结束进程")
                os._exit(0)  # 直接结束进程
        return False

    def increment_jobs_processed(self, count=1):
        """增加已处理的任务数"""
        self.jobs_processed += count
        
    def add_detailed_result(self, request_id, conversation_id, send_time, ttft, total_time, tokens_in, tokens_out):
        """添加详细的请求结果记录并实时写入CSV文件"""
        if self.detailed_logs:
            # 计算相关指标
            ttft_time = send_time + ttft if ttft > 0 else None
            total_end_time = send_time + total_time if total_time > 0 else None
            tpot = total_time - ttft if ttft > 0 and total_time > 0 else None
            
            # 保存到内存中
            self.detailed_results[request_id] = {
                'conversation_id': conversation_id,
                'send_time': send_time,
                'ttft_time': ttft_time,
                'total_time': total_end_time,
                'tokens_in': tokens_in,
                'tokens_out': tokens_out,
                'ttft': ttft,
                'tpot': tpot
            }
            
            # 实时写入CSV文件
            if hasattr(self, 'csv_file') and not self.csv_file.closed:
                csv_line = f"{request_id},{conversation_id},{send_time},{ttft_time},{total_end_time},{tokens_in},{tokens_out},{ttft},{tpot}\n"
                self.csv_file.write(csv_line)
                self.csv_file.flush()  # 立即写入磁盘
    
    def save_detailed_results(self):
        """关闭CSV文件"""
        if self.detailed_logs and hasattr(self, 'csv_file') and not self.csv_file.closed:
            self.csv_file.close()
            logger.info(f"已关闭详细日志文件: {self.csv_filename}")

async def async_replay_loop(start_timestamp, start_time, ep_config: dict = None, replay_mode: str = "timestamp", 
                          target_qps: float = 1.0, round_duration: int = 60, max_rounds: int = None,
                          detailed_logs: bool = False, cv=0.0):
    """根据不同模式选择相应的重放方式"""
    try:
        client = await client_manager.get_client(ep_config)
        
        # 创建结果收集器并设置为全局变量
        global global_result_collector
        
        if replay_mode == "timestamp":
            result_collector = ResultCollector(ep_config, round_duration, max_rounds, detailed_logs)
            global_result_collector = result_collector
            await replay_by_timestamp(client, result_collector, start_timestamp, start_time, round_duration, max_rounds, detailed_logs)
        elif replay_mode == "qps":
            result_collector = ResultCollector(ep_config, round_duration, max_rounds, detailed_logs)
            global_result_collector = result_collector
            await replay_by_qps(client, result_collector, target_qps, round_duration, max_rounds, detailed_logs, cv)
        else:
            logger.error(f"Unknown replay mode: {replay_mode}")
            
    finally:
        # 清理资源
        await client_manager.cleanup()

async def replay_by_timestamp(client, result_collector, start_timestamp, start_time, round_duration, max_rounds=None, detailed_logs=False):
    """按原始时间戳重放请求"""
    current_second = None
    current_jobs = []
    
    try:
        while True:
            try:
                # 如果当前没有待处理的任务且队列不为空
                if not current_jobs and not job_queue.empty():
                    # 获取下一个任务
                    next_job = job_queue.get()
                    current_second = next_job.second_timestamp
                    current_jobs = [next_job]
                    
                    # 获取同一秒的所有任务
                    while not job_queue.empty():
                        try:
                            peek_job = job_queue.queue[0]
                            # 如果属于当前这一秒，加入到批次中
                            if peek_job.second_timestamp == current_second:
                                job = job_queue.get()
                                current_jobs.append(job)
                            else:
                                # 下一个任务属于未来的时间，跳出循环
                                break
                        except IndexError:
                            # 队列可能被其他线程清空
                            break
                    
                    # 计算需要等待的时间以匹配原始时间戳
                    if start_timestamp is not None:
                        second_offset = current_second - start_timestamp
                        current_offset = time.perf_counter() - start_time
                        
                        # 如果需要等待以保持原始时间间隔
                        if second_offset > current_offset:
                            sleep_time = second_offset - current_offset
                            logger.debug(f"Sleeping for {sleep_time:.6f} seconds to maintain timing")
                            await asyncio.sleep(sleep_time)
                    
                    # 批量发送请求
                    batch_size = len(current_jobs)
                    logger.info(f"Sending batch of {batch_size} requests for second {current_second}")
                    
                    # 为每个任务创建异步任务并设置回调
                    for job in current_jobs:
                        send_time = time.perf_counter()
                        task = asyncio.create_task(send_request(client, job))
                        
                        # 添加回调函数
                        def callback(task, job_request_id=job.request_id, job_send_time=send_time, job=job):
                            try:
                                result = task.result()
                                request_id, status, ttft, total_time, tokens_in, tokens_out, error = result
                                result_collector.results_queue.put(result)
                                result_collector.total_requests += 1
                                if status == "OK":
                                    result_collector.successful_requests += 1
                                
                                # 添加详细日志数据
                                if detailed_logs and status == "OK":
                                    result_collector.add_detailed_result(
                                        request_id, job.conversation_id, job_send_time, ttft, total_time, tokens_in, tokens_out
                                    )
                            except Exception as e:
                                logger.error(f"Error in send_request callback: {e}")
                        
                        task.add_done_callback(callback)
                    
                    result_collector.increment_jobs_processed(batch_size)
                    
                    # 清空当前批次
                    current_jobs = []
                
                # 收集和处理结果
                result_collector.collect_results()
                if result_collector.check_and_report_metrics():
                    break
                
                # 如果没有新任务，短暂等待
                if job_queue.empty():
                    await asyncio.sleep(0.05)
                
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                await asyncio.sleep(0.1)
                continue
            
            # 检查是否应该退出
            if job_queue.empty() and not current_jobs:
                await asyncio.sleep(0.1)  # 给一个机会让更多任务进来
                if job_queue.empty():  # 二次确认
                    # 最后一次收集结果
                    result_collector.collect_results()
                    result_collector.check_and_report_metrics()
                    
                    # 保存详细日志结果
                    if detailed_logs:
                        result_collector.save_detailed_results()
                        
                    logger.info("All requests processed, exiting")
                    break
                    
    except Exception as e:
        logger.error(f"Error in timestamp replay: {e}")
        raise

async def replay_by_qps(client, result_collector, target_qps, round_duration,
                         max_rounds=None, detailed_logs=False, cv=0.0):
    """按指定QPS重放请求"""

    mean_interval = 1.0 / target_qps  # 平均请求间隔
    last_request_time = time.perf_counter()
    
    try:
        while True:
            try:
                if job_queue.empty():
                    await asyncio.sleep(0.01)
                    continue

                # 动态生成每个间隔
                if cv != 0.0 :
                    k = 1 / (cv ** 2)
                    theta = mean_interval / k
                    interval = max(0, np.random.gamma(k, theta))
                else:
                    interval = mean_interval
                
                current_time = time.perf_counter()
                if current_time - last_request_time < interval:
                    await asyncio.sleep(interval - (current_time - last_request_time))

                # no sleep
                send_time = time.perf_counter()
                last_request_time = send_time
                job = job_queue.get()
                task = asyncio.create_task(send_request(client, job))
                
                # 添加回调函数
                def callback(task, job_request_id=job.request_id, job_send_time=send_time, job=job):
                    try:
                        result = task.result()
                        request_id, status, ttft, total_time, tokens_in, tokens_out, error = result
                        result_collector.results_queue.put(result)
                        result_collector.total_requests += 1
                        if status == "OK":
                            result_collector.successful_requests += 1
                        
                        # 添加详细日志数据
                        if detailed_logs and status == "OK":
                            result_collector.add_detailed_result(
                                request_id, job.conversation_id, job_send_time, ttft, total_time, tokens_in, tokens_out
                            )
                    except Exception as e:
                        logger.error(f"Error in send_request callback: {e}")
                
                task.add_done_callback(callback)
                
                result_collector.increment_jobs_processed()
                    
                # 收集结果并报告指标
                result_collector.collect_results()
                if result_collector.check_and_report_metrics(qps=target_qps):
                    break

                # 精确控制循环间隔，避免CPU空转
                # next_decision_time = result_collector.round_start_time + ((result_collector.jobs_processed + 1) / target_qps)
                # sleep_time = max(0, next_decision_time - time.perf_counter())
                # if sleep_time > 0:
                #     await asyncio.sleep(sleep_time)
                # else:
                #     # 防止过度占用CPU
                #     await asyncio.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error in QPS replay: {e}")
                await asyncio.sleep(0.1)
                continue
            
            # 检查是否应该退出
            if job_queue.empty():
                await asyncio.sleep(0.1)  # 给一个机会让更多任务进来
                if job_queue.empty():  # 二次确认
                    # 最后一次收集结果
                    result_collector.collect_results()
                    result_collector.check_and_report_metrics(qps=target_qps)
                    
                    # 保存详细日志结果
                    if detailed_logs:
                        result_collector.save_detailed_results()
                        
                    logger.info("All requests processed, exiting")
                    break
                    
    except Exception as e:
        logger.error(f"Error in QPS replay: {e}")
        raise

def replay_thread(ep_config: dict = None, replay_mode: str = "timestamp", 
                 target_qps: float = 1.0, round_duration: int = 60, max_rounds: int = None,
                 detailed_logs: bool = False, cv = 0.0):
    """Thread B: Consume jobs from the queue and send requests in batches by second."""
    try:
        logger.info("Starting replay thread")
        
        # Get the first job to establish the start time
        if job_queue.empty():
            logger.error("Job queue is empty, cannot start replay")
            return
            
        first_job = job_queue.get()
        start_timestamp = first_job.second_timestamp  # Use second-level timestamp
        start_time = time.perf_counter()
        
        # Put the job back in the queue
        job_queue.put(first_job)
        
        # Create and run event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(async_replay_loop(
                start_timestamp, start_time, ep_config,
                replay_mode, target_qps, round_duration, max_rounds,
                detailed_logs, cv
            ))
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, stopping replay")
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error in replay thread: {e}")
        raise

def results_analysis(
    query_results, elts, model, concur_requests=None, qps=None, actual_qps=None, json_output=None):
    """分析请求结果并输出性能指标"""
    print("-------------------------")
    if json_output:
        try:
            json_output_f = open(json_output, "a+")
        except IOError as e:
            logger.error(f"无法创建或打开输出文件 {json_output}: {e}")
            json_output = None

    # 根据返回结果的格式调整列名
    if query_results and len(query_results[0]) > 6:  # 新格式包含request_id
        df = pd.DataFrame(
            query_results,
            columns=[
                "request_id",
                "valid",
                "ttft",
                "total_time",
                "tokens_in",
                "tokens_out",
                "cause",
            ],
        )
    else:
        df = pd.DataFrame(
            query_results,
            columns=[
                "valid",
                "ttft",
                "total_time",
                "tokens_in",
                "tokens_out",
                "cause",
            ],
        )
    
    # 计算成功率
    total_requests = len(df)
    successful_requests = len(df[df.valid == "OK"])
    success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
    
    cdf = df[df.valid != "Exception"].copy()
    if len(cdf) > 0:
        console = Console()
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric")
        table.add_column("Min")
        table.add_column("P50")
        table.add_column("P90")
        table.add_column("P95")
        table.add_column("P99")
        table.add_column("Max")

        if json_output:
            json_record = {}

        cdf["tokens_per_s"] = cdf.tokens_out / cdf.total_time
        mean_tokens_in = int(cdf["tokens_in"].mean())
        mean_tokens_out = int(cdf["tokens_out"].mean())

        s_per_output_token = (cdf["total_time"] - cdf["ttft"]) / (cdf["tokens_out"] - 1)

        total_input_tokens = cdf['tokens_in'].sum()
        total_output_tokens = cdf['tokens_out'].sum()
        
        total_time_minutes = max(elts) / 60  # 使用最大时间作为总运行时间
        
        # 如果是按QPS模式运行，可以用请求数量和QPS来估算时间
        if qps is not None:
            estimated_time_minutes = len(cdf) / (qps * 60)
            total_time_minutes = min(total_time_minutes, estimated_time_minutes)
            
        input_tokens_per_minute = total_input_tokens / total_time_minutes
        output_tokens_per_minute = total_output_tokens / total_time_minutes

        # 计算 SLO 达成率（SLO Attainment）。args.e2e_slo 为 float（秒）或 None
        slo_seconds = args.e2e_slo if hasattr(args, "e2e_slo") else None

        slo_attainment = None
        if slo_seconds is not None and total_requests > 0:
            # 定义：满足 valid=="OK" 且 total_time<=SLO 的请求占总请求的百分比
            met = ((df.valid == "OK") & (df.total_time <= slo_seconds)).sum()
            slo_attainment = (met / total_requests) * 100.0
 
        # 读取可选的 TTFT 和 TPOT SLO（单位：ms，int）并计算达成率
        ttft_slo_ms = args.ttft_slo if hasattr(args, "ttft_slo") else None
        tpot_slo_ms = args.tpot_slo if hasattr(args, "tpot_slo") else None

        ttft_slo_attainment = None
        tpot_slo_attainment = None

        if ttft_slo_ms is not None and total_requests > 0:
            met_ttft = ((df.valid == "OK") & (df.ttft <= (ttft_slo_ms / 1000.0))).sum()
            ttft_slo_attainment = (met_ttft / total_requests) * 100.0

        if tpot_slo_ms is not None and total_requests > 0:
            # 使 TPOT 达成率逻辑与 TTFT 一致：
            # 百分比= 总请求中 满足 valid=="OK" 且 TPOT<=阈值 的比例
            tpot_ms_all = ((df["total_time"] - df["ttft"]) / (df["tokens_out"] - 1)) * 1000
            valid_mask = (df["valid"] == "OK") & (df["tokens_out"] > 1)
            met_tpot = (valid_mask & (tpot_ms_all <= tpot_slo_ms)).sum()
            tpot_slo_attainment = (met_tpot / total_requests) * 100.0

    title = f"{model}\n("
    if concur_requests is not None:
        title += f"concurrency={concur_requests}, "
    if qps is not None:
        title += f"target_qps={int(qps) if int(qps) == qps else qps}, "
    if actual_qps is not None:
        title += f"actual_qps={actual_qps:.2f}, "
    title += f"success_rate={success_rate:.2f}%, "
    title += f"input_tokens={mean_tokens_in}, output_tokens={mean_tokens_out})"

    # 将 SLO 指标与达成率移动到标题的新一行中
    if (hasattr(args, "ttft_slo") and args.ttft_slo is not None) or \
        (hasattr(args, "tpot_slo") and args.tpot_slo is not None) or \
        (hasattr(args, "e2e_slo") and args.e2e_slo is not None):
        line_parts = []
        if 'slo_attainment' in locals() and slo_attainment is not None:
            line_parts.append(f"e2e_slo_attainment: {slo_attainment:.2f}%")
        if 'ttft_slo_attainment' in locals() and ttft_slo_attainment is not None:
            line_parts.append(f"ttft_slo_attainment: {ttft_slo_attainment:.2f}%")
        if 'tpot_slo_attainment' in locals() and tpot_slo_attainment is not None:
            line_parts.append(f"tpot_slo_attainment: {tpot_slo_attainment:.2f}%")
        if line_parts:
            title = title + "\n" + ", ".join(line_parts)
 
    table.title = title

    if json_output:
        if concur_requests is not None:
            json_record["concurrency"] = concur_requests
        if qps is not None:
            json_record["target_qps"] = qps
        if actual_qps is not None:
            json_record["actual_qps"] = actual_qps
        json_record["success_rate"] = success_rate
        json_record["input_tokens"] = mean_tokens_in
        json_record["output_tokens"] = mean_tokens_out
        json_record["model"] = model
        json_record["input_tokens_per_minute"] = input_tokens_per_minute
        json_record["output_tokens_per_minute"] = output_tokens_per_minute
        if slo_seconds is not None and slo_attainment is not None:
            json_record["slo_seconds"] = slo_seconds
            json_record["slo_attainment"] = slo_attainment
        # 追加 TTFT/TPOT SLO 输出
        if 'ttft_slo_ms' not in locals():
            ttft_slo_ms = args.ttft_slo if hasattr(args, "ttft_slo") else None
        if 'tpot_slo_ms' not in locals():
            tpot_slo_ms = args.tpot_slo if hasattr(args, "tpot_slo") else None
        if 'ttft_slo_attainment' in locals() and ttft_slo_ms is not None and ttft_slo_attainment is not None:
            json_record["ttft_slo_ms"] = ttft_slo_ms
            json_record["ttft_slo_attainment"] = ttft_slo_attainment
        if 'tpot_slo_attainment' in locals() and tpot_slo_ms is not None and tpot_slo_attainment is not None:
            json_record["tpot_slo_ms"] = tpot_slo_ms
            json_record["tpot_slo_attainment"] = tpot_slo_attainment

    def show_metric(name, unit, val):
        table.add_row(
            f"{name}({unit})",
            f"{val.min():.3f}",
            f"{val.quantile(0.5):.3f}",
            f"{val.quantile(0.9):.3f}",
            f"{val.quantile(0.95):.3f}",
            f"{val.quantile(0.99):.3f}",
            f"{val.max():.3f}",
        )
        if json_output:
            json_record[name] = {
                "unit": unit,
                "min": val.min(),
                "p50": val.quantile(0.5),
                "p90": val.quantile(0.9),
                "p95": val.quantile(0.95),
                "p99": val.quantile(0.99),
                "max": val.max(),
            }

    show_metric("Latency", "s", cdf["total_time"])
    show_metric("Throughput", "tokens/s", cdf["tokens_per_s"])
    show_metric("TTFT", "s", cdf["ttft"])
    show_metric("TPOT", "ms", s_per_output_token * 1000)
    show_metric("Input Tokens per Minute", "tokens/min", pd.Series([input_tokens_per_minute]))
    show_metric("Output Tokens per Minute", "tokens/min", pd.Series([output_tokens_per_minute]))

    console.print(table)

    def error_analysis(df):
        exceptions = df[df.valid == "Exception"]
        exceptions_by_cause = Counter()
        for cause in exceptions["cause"]:
            exceptions_by_cause[cause] += 1

        if exceptions_by_cause:
            print("\nExceptions by cause:")
            for cause, count in exceptions_by_cause.items():
                print(f" - {count}: {cause}")

            if json_output:
                json_record["exceptions"] = {}
                for cause, count in exceptions_by_cause.items():
                    json_record["exceptions"][cause] = count

    error_analysis(df)
    print("-------------------------")

    if json_output:
        json.dump(json_record, json_output_f)
        json_output_f.write("\n")
        json_output_f.close()


def main(args, sample_start, sample_end):
    """Main function to start both threads."""
    try:
        # 设置日志级别
        set_logging_level(args.verbose)
        
        # 创建配置
        ep_config = {
            "api_base": args.api_base,
            "api_key": args.api_key,
            "model": args.model,
            "use_chat": args.use_chat,
            "max_tokens": args.max_tokens,
            "dataset": args.dataset
        }
        
        # 创建全局结果收集器，用于在程序退出时保存结果
        global global_result_collector
        global_result_collector = None
        
        # 注册全局信号处理函数
        def global_signal_handler(sig, frame):
            logger.info("收到中断信号，正在优雅退出...")
            if global_result_collector and args.detailed_logs:
                logger.info("关闭详细日志文件...")
                global_result_collector.save_detailed_results()
            logger.info("程序退出")
            sys.exit(0)
        
        # 注册SIGINT信号处理函数（只在主线程中注册）
        if args.detailed_logs:
            signal.signal(signal.SIGINT, global_signal_handler)
        
        # Start the log reader thread
        reader_thread = threading.Thread(target=log_reader_thread,
                                args=(args.input, args.preload_time, sample_start, sample_end, ep_config, args.dataset))
        reader_thread.daemon = True
        reader_thread.start()
        
        # Wait for the preload time
        logger.info(f"Waiting {args.preload_time} seconds for log reader to preload jobs")
        time.sleep(args.preload_time)
        
        # Start the replay thread with ep_config and replay parameters
        replay_thread_instance = threading.Thread(target=replay_thread, 
                                                args=(ep_config, args.replay_mode, args.target_qps, args.round_duration,
                                                      args.max_rounds, args.detailed_logs, args.cv))
        replay_thread_instance.daemon = True
        replay_thread_instance.start()
        
        try:
            # Wait for both threads to complete
            reader_thread.join()
            replay_thread_instance.join()
        finally:
            # 确保在主程序退出时清理资源
            loop = asyncio.get_event_loop()
            loop.run_until_complete(client_manager.cleanup())
            
            # 关闭详细日志文件
            if global_result_collector and args.detailed_logs:
                global_result_collector.save_detailed_results()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, stopping threads")
        # 在键盘中断时关闭详细日志文件
        if global_result_collector and args.detailed_logs:
            logger.info("关闭详细日志文件...")
            global_result_collector.save_detailed_results()
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Replay chat log requests')
    parser.add_argument('--input', '-i', default="/mnt/shared/data/replay-logs-origin.log",
                        help='Input log file (default: chat_log.log)')
    parser.add_argument('--preload-time', '-p', type=int, default=20,
                        help='Preload time in seconds (default: 2)')
    parser.add_argument("--api-key", type=str, default="a" * 32,
                        help="API key")
    parser.add_argument("--api-base", type=str, default="http://localhost:8080/api/v1",
                        help="API base url")
    parser.add_argument("--model", type=str, default="Nitral-AI/Captain-Eris_Violet-V0.420-12B",
                        help="Model name to use")
    parser.add_argument("--use-chat", type=bool, default=False,
                        help="Whether to use the chat endpoint")
    parser.add_argument("--max-tokens", type=int, default=180,
                        help="Maximum number of tokens to generate (default: 180)")
    parser.add_argument("--round-duration", type=int, default=60,
                        help="Duration of each round in seconds (default: 60)")
    parser.add_argument("--max-rounds", type=int, default=None,
                        help="Maximum number of rounds to run (default: None, run until all requests are processed)")

    # 选择重放模式
    parser.add_argument("--replay-mode", type=str, choices=["timestamp", "qps"],
                        default="timestamp", help="Replay mode: timestamp/qps")
    parser.add_argument('--sample-range', type=float, nargs=2, default=[0.0, 1.0],
                        metavar=('START', 'END'),
                        help='Sample range [START, END) to control the percentage of requests to send (e.g., 0.0 0.2). Default: [0.0, 1.0]')
    parser.add_argument("--target-qps", type=float, default=1.0,
                        help="Target QPS for qps mode")
    # E2E SLO：端到端延迟目标（单位：秒，float）。例如：--e2e-slo 5.0
    parser.add_argument("--e2e-slo", type=float, default=None,
                        help="End-to-end latency SLO in seconds (float). Example: 5.0. If set, will report E2E SLO Attainment")
    # 新增：TTFT/TPOT 的 SLO（单位：毫秒，int）
    parser.add_argument("--ttft-slo", type=int, default=None,
                        help="TTFT SLO in milliseconds (int). If set, will report TTFT SLO Attainment")
    parser.add_argument("--tpot-slo", type=int, default=None,
                        help="TPOT SLO in milliseconds (int). If set, will report TPOT SLO Attainment")

    # 是否打印详细日志
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    # 是否将结果输出到json已经选择输出的路径
    parser.add_argument("--json-output", type=str, default=None,
                        help="If set, the file to save the results in json format")
    # 是否记录详细请求数据并导出CSV
    parser.add_argument("--detailed-logs", type=str, default=False,
                        help="Enable detailed logging of each request with request-id, timestamps and token counts. Optionally specify a path to save the CSV file, otherwise default path will be used")
    
    parser.add_argument("--dataset", type=str, default="shibing624/sharegpt_gpt4",
                        help="Use shibing624/sharegpt_gpt4 dataset as input instead of log file")
    
    parser.add_argument("--cv", type=float, default=0.0,
                        help="Coefficient of Variation (CV) to control the unevenness of request intervals. Default: 0.0") 
    args = parser.parse_args()
    
    # 确保采样率在合理范围
    sample_start, sample_end = args.sample_range
    if not (0.0 <= sample_start < sample_end <= 1.0):
        raise ValueError(f"Invalid sample range [{sample_start}, {sample_end}). Must be 0.0 <= START < END <= 1.0")
    logger.info(f"Using sample range: [{sample_start}, {sample_end})")
    
    try:
        main(args, sample_start, sample_end)
    except Exception as e:
        logger.error(f"Failed to process log file: {e}") 