import json
import re
import time
import threading
import queue
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

def should_process_conversation(conversation_id: str, sample_rate: float) -> bool:
    """基于conversationId确定是否处理该请求，根据采样率进行筛选。"""
    if sample_rate >= 1.0:
        return True
    if sample_rate <= 0.0:
        return False
    
    # 计算hash值，使其分布在0-1之间
    hash_obj = hashlib.md5(conversation_id.encode())
    hash_bytes = hash_obj.digest()
    # 取前4个字节转换为整数，然后归一化到0-1范围
    hash_int = int.from_bytes(hash_bytes[:4], byteorder='big')
    normalized_hash = hash_int / (2**32)
    
    # 如果哈希值小于采样率，就处理它
    return normalized_hash < sample_rate

def process_log_line(line: str, sample_rate: float = 1.0, ep_config: dict = None) -> Optional[ReplayJob]:
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
        if not should_process_conversation(conversation_id, sample_rate):
            logger.debug(f"Skipping conversation {conversation_id} due to sampling")
            return None

        # 将conversation_id由uuid转为int, 不然production-stack的router会报错
        hash_obj = hashlib.sha256(conversation_id.encode())
        conversation_id_int = int(hash_obj.hexdigest(), 16)
        conversation_id = conversation_id_int

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
            # print("messages", messages)
            # print("---------------------------------------------------------------")
            # 确保messages不为空
            if not messages:
                logger.warning(f"Empty messages for conversation {conversation_id}, skipping")
                return None
            
            body = {
                "model": ep_config["model"],
                "messages": messages,
                "stream": True,
                "max_tokens": ep_config.get("max_tokens", 200),
                "temperature": 0
            }
            url = f"{ep_config['api_base'].rstrip('/')}/chat/completions"
        else:
            # 对于非chat模式，构造一个包含单个消息的messages数组
            body = {
                "model": ep_config["model"],
                "messages": [request_data['body'].get('prompt', '')],
                "stream": True,
                "max_tokens": ep_config.get("max_tokens", 200),
                "temperature": 0
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

def log_reader_thread(input_file: str, preload_time: int = 180, sample_rate: float = 1.0, ep_config: dict = None):
    """Thread A: Read log file and add jobs to the queue."""
    try:
        logger.info(f"Starting log reader thread with {preload_time} seconds preload time and {sample_rate*100:.1f}% sample rate")
        job_count = 0
        skipped_count = 0
        last_timestamp = None
        
        with open(input_file, 'r') as fin:
            for line_number, line in enumerate(fin, 1):
                try:
                    logger.debug(f"Processing line {line_number}")
                    job = process_log_line(line.strip(), sample_rate, ep_config)
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

async def send_request(client, job):
    """Send a single request asynchronously and collect metrics."""
    try:
        start_time = time.time()
        ttft = None
        tokens_in = 0
        tokens_out = 0
        
        # Add conversation_id to headers
        extra_headers = {"x-user-id": str(job.conversation_id)} if job.conversation_id else {}
        
        if job.use_chat:
            if not job.body.get("messages"):
                logger.warning("Empty messages array, skipping request")
                return ("Exception", -1, -1, -1, -1, "Empty messages")
            
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
            )
            
        words = ""
        async for tok in response:
            if not tok.choices:
                continue
            if job.use_chat:
                delta = tok.choices[0].delta
                if delta.content:
                    if ttft is None:
                        ttft = time.time() - start_time
                    words += delta.content
            else:
                delta = tok.choices[0]
                if delta.text:
                    if ttft is None:
                        ttft = time.time() - start_time
                    words += delta.text
                    
        tokens_in = tok.usage.prompt_tokens
        tokens_out = tok.usage.completion_tokens
        total_time = time.time() - start_time
        
        return ("OK", ttft, total_time, tokens_in, tokens_out, "")
        
    except asyncio.TimeoutError:
        logger.debug("Request timed out after 2s")
        return ("Exception", -1, -1, -1, -1, "Timeout")
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return ("Exception", -1, -1, -1, -1, str(e))

# 全局会话对象，避免频繁创建和销毁
global_client = None
background_tasks = set()

async def endpoint_evaluation_request(client, ep_config):
    """从日志中提取请求并发送，收集性能指标"""
    try:
        # 从队列中获取一个请求
        if job_queue.empty():
            return ("Exception", -1, -1, -1, -1, "No more requests in queue")
            
        job = job_queue.get()
        return await send_request(client, job)
        
    except Exception as e:
        logger.error(f"Request failed in endpoint evaluation: {e}")
        return ("Exception", -1, -1, -1, -1, str(e))

async def send_batch_requests_without_waiting(jobs, ep_config: dict = None):
    """发送批量请求但不等待完成。"""
    global global_client
    
    if global_client is None:
        if ep_config is None:
            raise ValueError("ep_config must be provided when creating new session")
        global_client = openai.AsyncOpenAI(
            base_url=ep_config["api_base"],
            api_key=ep_config["api_key"]
        )
    
    # 创建背景任务来处理响应
    async def process_responses(jobs_list):
        try:
            tasks = []
            for job in jobs_list:
                task = asyncio.create_task(send_request(global_client, job))
                tasks.append(task)
            
            # 等待所有请求完成并记录结果
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for result in results if result[0] == "OK")
            logger.info(f"Sent batch of {len(jobs_list)} requests, {success_count} successful")
            return results
        except Exception as e:
            logger.error(f"Error in background task: {e}")
            return []
    
    # 创建后台任务并保持引用防止被垃圾回收
    task = asyncio.create_task(process_responses(jobs))
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)
    
    # 等待任务完成并返回结果
    return await task

def group_jobs_by_second(jobs):
    """Group jobs by their second timestamp."""
    jobs.sort(key=lambda job: job.second_timestamp)
    return {k: list(g) for k, g in groupby(jobs, key=lambda job: job.second_timestamp)}

class ResultCollector:
    """用于收集和分析请求结果的类"""
    def __init__(self, ep_config, round_duration):
        self.results_queue = queue.Queue()
        self.query_results = []
        self.elts = []
        self.jobs_processed = 0
        self.round_start_time = time.time()
        self.ep_config = ep_config
        self.round_duration = round_duration
        self.total_requests = 0
        self.successful_requests = 0

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
        current_time = time.time()
        if current_time - self.round_start_time >= self.round_duration:
            elapsed_time = current_time - self.round_start_time
            self.elts.append(elapsed_time)
            actual_qps = self.jobs_processed / elapsed_time
            
            # 分析结果
            results_analysis(
                self.query_results,
                self.elts,
                self.ep_config["model"],
                qps=qps or actual_qps,
                concur_requests=concur_requests,
                json_output=args.json_output,
            )
            
            # 重置统计
            self.query_results = []
            self.jobs_processed = 0
            self.round_start_time = current_time

    def increment_jobs_processed(self, count=1):
        """增加已处理的任务数"""
        self.jobs_processed += count

async def replay_by_timestamp(client, ep_config, start_timestamp, start_time, round_duration):
    """按原始时间戳重放请求"""
    result_collector = ResultCollector(ep_config, round_duration)
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
                        current_offset = time.time() - start_time
                        
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
                        task = asyncio.create_task(send_request(client, job))
                        task.add_done_callback(result_collector.task_done_callback)
                    
                    result_collector.increment_jobs_processed(batch_size)
                    
                    # 清空当前批次
                    current_jobs = []
                
                # 收集和处理结果
                result_collector.collect_results()
                result_collector.check_and_report_metrics()
                
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
                    break
                    
    except Exception as e:
        logger.error(f"Error in timestamp replay: {e}")
        raise

async def replay_by_qps(client, ep_config, target_qps, round_duration):
    """按指定QPS重放请求"""
    result_collector = ResultCollector(ep_config, round_duration)
    time_between_requests = 1.0 / target_qps
    
    try:
        while True:
            try:
                if job_queue.empty():
                    await asyncio.sleep(0.1)
                    continue
                
                request_start = time.time()
                job = job_queue.get()
                
                # 创建异步任务并设置回调
                task = asyncio.create_task(send_request(client, job))
                task.add_done_callback(result_collector.task_done_callback)
                
                result_collector.increment_jobs_processed()
                
                # 收集和处理结果
                result_collector.collect_results()
                result_collector.check_and_report_metrics(qps=target_qps)
                
                # 计算需要等待的时间以维持目标QPS
                elapsed = time.time() - request_start
                if elapsed < time_between_requests:
                    await asyncio.sleep(time_between_requests - elapsed)
                
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
                    break
                    
    except Exception as e:
        logger.error(f"Error in QPS replay: {e}")
        raise

async def async_replay_loop(start_timestamp, start_time, ep_config: dict = None, replay_mode: str = "timestamp", 
                          target_qps: float = 1.0, round_duration: int = 60):
    """根据不同模式选择相应的重放方式"""
    try:
        client = openai.AsyncOpenAI(
            base_url=ep_config["api_base"],
            api_key=ep_config["api_key"]
        )
        
        if replay_mode == "timestamp":
            await replay_by_timestamp(client, ep_config, start_timestamp, start_time, round_duration)
        elif replay_mode == "qps":
            await replay_by_qps(client, ep_config, target_qps, round_duration)
        else:
            logger.error(f"Unknown replay mode: {replay_mode}")
            
    finally:
        # 清理全局会话
        global global_client
        if global_client:
            await global_client.close()
            global_client = None

def replay_thread(ep_config: dict = None, replay_mode: str = "timestamp", 
                 target_qps: float = 1.0, round_duration: int = 60):
    """Thread B: Consume jobs from the queue and send requests in batches by second."""
    try:
        logger.info("Starting replay thread")
        
        # Get the first job to establish the start time
        if job_queue.empty():
            logger.error("Job queue is empty, cannot start replay")
            return
            
        first_job = job_queue.get()
        start_timestamp = first_job.second_timestamp  # Use second-level timestamp
        start_time = time.time()
        
        # Put the job back in the queue
        job_queue.put(first_job)
        
        # Create and run event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(async_replay_loop(
                start_timestamp, start_time, ep_config,
                replay_mode, target_qps, round_duration
            ))
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, stopping replay")
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error in replay thread: {e}")
        raise

def results_analysis(query_results, elts, model, concur_requests=None, qps=None, json_output=None):
    """分析请求结果并输出性能指标"""
    print("-------------------------")
    if json_output:
        json_output_f = open(json_output, "a")

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

        title = f"{model}\n("
        if concur_requests is not None:
            title += f"concurrency={concur_requests}, "
        if qps is not None:
            title += f"qps={int(qps) if int(qps) == qps else qps}, "
        title += f"success_rate={success_rate:.2f}%, "
        title += f"input_tokens={mean_tokens_in}, output_tokens={mean_tokens_out})"
        table.title = title

        if json_output:
            if concur_requests is not None:
                json_record["concurrency"] = concur_requests
            if qps is not None:
                json_record["qps"] = qps
            json_record["success_rate"] = success_rate
            json_record["input_tokens"] = mean_tokens_in
            json_record["output_tokens"] = mean_tokens_out
            json_record["model"] = model
            json_record["input_tokens_per_minute"] = input_tokens_per_minute
            json_record["output_tokens_per_minute"] = output_tokens_per_minute

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
        show_metric("Success Rate", "%", pd.Series([success_rate]))

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

async def endpoint_evaluation_round(client, concur_requests, ep_config):
    """执行一轮并发请求评估"""
    results = await asyncio.gather(*(
        endpoint_evaluation_request(client, ep_config) for _ in range(concur_requests)
    ))
    return results

def endpoint_evaluation_qps(client, ep_config, results_queue, stop_event):
    """按QPS执行请求评估"""
    loop = asyncio.new_event_loop()

    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    threading.Thread(target=run_loop).start()

    time_between_requests = 1 / args.qps

    def task_done_callback(task):
        results_queue.put(task.result())

    while True:
        if stop_event.is_set():
            print("stop event received, stopping loop")
            loop.call_soon_threadsafe(loop.stop)
            return

        st = time.time()
        future = asyncio.run_coroutine_threadsafe(
            endpoint_evaluation_request(client, ep_config), loop
        )
        future.add_done_callback(task_done_callback)
        et = time.time()
        tosleep = time_between_requests - (et - st)
        if tosleep > 0:
            time.sleep(tosleep)

    return results_queue

def endpoint_evaluation(ep_config):
    """执行端点评估"""
    client = openai.AsyncOpenAI(
        base_url=ep_config["api_base"], api_key=ep_config["api_key"]
    )
    loop = asyncio.new_event_loop()

    if ep_config["model"] is None:
        async def get_model():
            async for model in client.models.list():
                ep_config["model"] = model.id
                break
        loop.run_until_complete(get_model())

    for _ in range(args.warmup):
        loop.run_until_complete(endpoint_evaluation_request(client, ep_config))

    if args.qps is not None:
        num_results_per_round = math.ceil(args.qps * args.round_duration)
        query_results = []
        elts = []
        results_queue = queue.Queue()
        stop_event = threading.Event()
        threading.Thread(
            target=endpoint_evaluation_qps,
            args=(client, ep_config, results_queue, stop_event),
        ).start()

        st = time.time()
        try:
            while True:
                round_results = []
                round_start = time.time()
                while time.time() - round_start < args.round_duration:
                    try:
                        result = results_queue.get(timeout=0.1)
                        round_results.append(result)
                    except queue.Empty:
                        pass
                query_results.extend(round_results)
                et = time.time()
                elts.append(et - st)
                st = et
                results_analysis(
                    query_results,
                    elts,
                    ep_config["model"],
                    qps=args.qps,
                    json_output=args.json_output,
                )
                query_results = []
                elts = []
        finally:
            stop_event.set()
    else:
        for concur_requests in args.concur_requests:
            query_results = []
            elts = []
            for _ in range(args.rounds):
                st = time.time()
                results = []
                while time.time() - st < args.round_duration:
                    round_results = loop.run_until_complete(
                        endpoint_evaluation_round(client, concur_requests, ep_config)
                    )
                    results.extend(round_results)
                query_results.extend(results)
                et = time.time()
                elt = et - st
                elts.append(elt)
            results_analysis(
                query_results,
                elts,
                ep_config["model"],
                concur_requests,
                json_output=args.json_output,
            )

def main(args):
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
            "max_tokens": args.max_tokens
        }
        
        # Start the log reader thread
        reader_thread = threading.Thread(target=log_reader_thread, args=(args.input, args.preload_time, args.sample_rate, ep_config))
        reader_thread.daemon = True
        reader_thread.start()
        
        # Wait for the preload time
        logger.info(f"Waiting {args.preload_time} seconds for log reader to preload jobs")
        time.sleep(args.preload_time)
        
        # Start the replay thread with ep_config and replay parameters
        replay_thread_instance = threading.Thread(
            target=replay_thread, 
            args=(ep_config, args.replay_mode, args.target_qps, args.round_duration)
        )
        replay_thread_instance.daemon = True
        replay_thread_instance.start()
        
        # Wait for both threads to complete
        reader_thread.join()
        replay_thread_instance.join()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, stopping threads")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Replay chat log requests')
    parser.add_argument('--input', '-i', default="/mnt/shared/data/replay-logs-origin.log",
                        help='Input log file (default: chat_log.log)')
    parser.add_argument('--preload-time', '-p', type=int, default=2,
                        help='Preload time in seconds (default: 2)')
    parser.add_argument("--api-key", type=str, default="a" * 32,
                        help="API key")
    parser.add_argument("--api-base", type=str, default="http://localhost:8080/api/v1",
                        help="API base url")
    parser.add_argument("--model", type=str, default="Nitral-AI/Captain-Eris_Violet-V0.420-12B",
                        help="Model name to use")
    parser.add_argument("--use-chat", type=bool, default=False,
                        help="Whether to use the chat endpoint")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Maximum number of tokens to generate (default: 200)")
    parser.add_argument("--round-duration", type=int, default=60,
                        help="Duration of each round in seconds (default: 60)")

    # 选择重放模式
    parser.add_argument("--replay-mode", type=str, choices=["timestamp", "qps"],
                        default="timestamp", help="Replay mode: timestamp/qps")
    parser.add_argument('--sample-rate', '-s', type=float, default=0.1,
                        help='Sample rate (0.0-1.0) to control the percentage of requests to send (default: 0.1)')
    parser.add_argument("--target-qps", type=float, default=1.0,
                        help="Target QPS for qps mode")
    
    # 是否打印详细日志
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    # 是否将结果输出到json已经选择输出的路径
    parser.add_argument("--json-output", type=str, default=None,
                        help="If set, the file to save the results in json format")
    
    args = parser.parse_args()
    
    # 确保采样率在合理范围
    args.sample_rate = max(0.0, min(1.0, args.sample_rate))
    
    try:
        main(args)
    except Exception as e:
        logger.error(f"Failed to process log file: {e}") 
