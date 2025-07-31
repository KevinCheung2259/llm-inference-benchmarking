#!/usr/bin/env python3
"""
基于Tokenizer的数据集脱敏转换脚本
使用指定模型的tokenizer进行分词，保留token ID分布，输出为JSON格式
"""

import json
import re
import hashlib
import argparse
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Set
import os
from collections import defaultdict
import random
from transformers import AutoTokenizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TokenizerBasedDesensitizer:
    """基于Tokenizer的数据集脱敏处理器"""
    
    def __init__(self, model_name: str = "Nitral-AI/Captain-Eris_Violet-V0.420-12B"):
        self.model_name = model_name
        
        # 加载tokenizer
        logger.info(f"正在加载模型tokenizer: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            logger.info("Tokenizer加载成功")
        except Exception as e:
            logger.error(f"加载tokenizer失败: {e}")
            logger.info("使用GPT-2作为备用tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # 映射关系
        self.token_id_mapping = {}  # 原始token_id -> 脱敏token_id
        self.conversation_id_mapping = {}
        
        # 统计信息
        self.processed_lines = 0
        self.skipped_lines = 0
        self.processed_conversations = set()
        
        # 获取词汇表大小
        self.vocab_size = len(self.tokenizer)
        logger.info(f"词汇表大小: {self.vocab_size}")
        
        # 创建token ID的脱敏映射
        self._create_token_id_mapping()
    
    def _create_token_id_mapping(self):
        """创建token ID的脱敏映射"""
        logger.info("正在创建token ID置换表...")
        
        # 生成与原词表等长的置换表
        original_ids = list(range(self.vocab_size))
        shuffled_ids = list(range(self.vocab_size))
        
        # 使用固定种子确保可复现性
        random.seed(42)
        random.shuffle(shuffled_ids)
        
        # 创建一对一的映射关系（置换表）
        for i, original_id in enumerate(original_ids):
            self.token_id_mapping[original_id] = shuffled_ids[i]
        
        logger.info(f"创建了 {len(self.token_id_mapping)} 个token ID映射的置换表")
    
    def _desensitize_text(self, text: str) -> List[int]:
        """对文本进行tokenizer分词并返回脱敏后的token IDs"""
        if not text:
            return []
        
        try:
            # 使用tokenizer进行分词
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            
            # 使用置换表对token IDs进行脱敏映射
            desensitized_ids = []
            for token_id in token_ids:
                # 由于置换表已经包含了所有可能的token ID，直接查找即可
                if token_id < self.vocab_size:
                    desensitized_ids.append(self.token_id_mapping[token_id])
                else:
                    # 如果token_id超出词汇表范围，保持原样或映射到未知token
                    logger.warning(f"Token ID {token_id} 超出词汇表范围 {self.vocab_size}")
                    desensitized_ids.append(token_id % self.vocab_size)  # 简单的取模处理
            
            return desensitized_ids
            
        except Exception as e:
            logger.error(f"分词失败: {e}")
            return []
    
    def _desensitize_conversation_id(self, conversation_id: str) -> str:
        """对会话ID进行脱敏"""
        if conversation_id in self.conversation_id_mapping:
            return self.conversation_id_mapping[conversation_id]
        
        # 生成新的会话ID，保持一定的格式
        hash_obj = hashlib.md5(conversation_id.encode('utf-8'))
        hex_hash = hash_obj.hexdigest()
        
        # 生成格式化的新ID
        new_id = f"conv_{hex_hash[:8]}_{hex_hash[8:16]}"
        self.conversation_id_mapping[conversation_id] = new_id
        
        return new_id
    
    def _desensitize_messages(self, messages: Any) -> Any:
        """对消息内容进行脱敏"""
        if isinstance(messages, str):
            # 如果是字符串，直接返回token IDs
            return self._desensitize_text(messages)
        elif isinstance(messages, list):
            desensitized_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    desensitized_msg = {}
                    for key, value in msg.items():
                        if key == 'content' and isinstance(value, str):
                            # 将文本内容转换为token IDs
                            desensitized_msg[key] = self._desensitize_text(value)
                        elif isinstance(value, str) and key in ['role', 'name']:
                            # 角色和名称保持不变
                            desensitized_msg[key] = value
                        else:
                            desensitized_msg[key] = value
                    desensitized_messages.append(desensitized_msg)
                elif isinstance(msg, str):
                    desensitized_messages.append(self._desensitize_text(msg))
                else:
                    desensitized_messages.append(msg)
            return desensitized_messages
        else:
            return messages
    
    def find_json_objects(self, text: str) -> list:
        """在文本中找到所有JSON对象"""
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
            # 查找所有JSON对象
            json_objects = self.find_json_objects(line)
            if not json_objects:
                return None
                
            # 尝试解析每个JSON对象
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
            logger.debug(f"解析JSON时出错: {e}")
            return None
    
    def process_log_line(self, line: str) -> Optional[Dict]:
        """处理单行日志并返回脱敏后的JSON对象"""
        try:
            # 提取时间戳
            timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)', line)
            if not timestamp_match:
                return None
            
            timestamp = timestamp_match.group(1)
            
            # 提取并解析JSON
            request_data = self.extract_json_from_log(line)
            if not request_data:
                return None
            
            # 获取原始数据
            original_conversation_id = request_data.get('conversationId', '')
            original_body = request_data.get('body', {})
            original_prompt = original_body.get('prompt', [])
            
            # 脱敏处理
            desensitized_conversation_id = self._desensitize_conversation_id(original_conversation_id)
            desensitized_prompt = self._desensitize_messages(original_prompt)
            
            # 构造脱敏后的数据对象
            desensitized_data = {
                'timestamp': timestamp,
                'conversation_id': desensitized_conversation_id,
                'messages': desensitized_prompt
            }
            
            # 统计
            self.processed_conversations.add(original_conversation_id)
            self.processed_lines += 1
            
            return desensitized_data
            
        except Exception as e:
            logger.debug(f"处理行时出错: {e}")
            self.skipped_lines += 1
            return None
    
    def process_file(self, input_file: str, output_file: str):
        """处理整个文件"""
        logger.info(f"开始处理文件: {input_file}")
        logger.info(f"输出文件: {output_file}")
        
        processed_count = 0
        
        # 流式处理：边读边写，节省内存
        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_file, 'w', encoding='utf-8') as fout:
            for line_number, line in enumerate(fin, 1):
                try:
                    desensitized_item = self.process_log_line(line.strip())
                    if desensitized_item:
                        # 立即写入JSONL文件
                        json_line = json.dumps(desensitized_item, ensure_ascii=False, separators=(',', ':'))
                        fout.write(json_line + '\n')
                        processed_count += 1
                    
                    if line_number % 1000 == 0:
                        logger.info(f"已处理 {line_number} 行，成功转换 {processed_count} 行")
                        
                except Exception as e:
                    logger.error(f"处理第 {line_number} 行时出错: {e}")
                    continue
        
        logger.info(f"处理完成!")
        logger.info(f"总行数: {line_number}")
        logger.info(f"成功处理行数: {processed_count}")
        logger.info(f"跳过行数: {line_number - processed_count}")
        logger.info(f"处理的会话数: {len(self.processed_conversations)}")
        logger.info(f"生成的token ID映射数: {len(self.token_id_mapping)}")
        logger.info(f"生成的会话ID映射数: {len(self.conversation_id_mapping)}")
    
    def save_mappings(self, mapping_file: str):
        """保存映射关系到文件"""
        logger.info(f"保存置换表到: {mapping_file}")
        
        mappings = {
            'model_name': self.model_name,
            'tokenizer_vocab_size': self.vocab_size,
            'shuffle_table': self.token_id_mapping,  # 完整的置换表
            'conversation_id_mapping': self.conversation_id_mapping,
            'statistics': {
                'processed_lines': self.processed_lines,
                'skipped_lines': self.skipped_lines,
                'processed_conversations': len(self.processed_conversations),
                'shuffle_table_size': len(self.token_id_mapping),
                'conversation_id_mappings': len(self.conversation_id_mapping)
            }
        }
        
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mappings, f, ensure_ascii=False, indent=2)
        
        logger.info("置换表保存完成")
    
    def decode_tokens(self, token_ids: List[int]) -> str:
        """将脱敏后的token IDs解码为文本（用于调试）"""
        try:
            # 创建反向置换表
            if not hasattr(self, '_reverse_mapping'):
                self._reverse_mapping = {v: k for k, v in self.token_id_mapping.items()}
            
            # 反向映射token IDs到原始IDs
            original_ids = []
            for token_id in token_ids:
                if token_id in self._reverse_mapping:
                    original_ids.append(self._reverse_mapping[token_id])
                else:
                    logger.warning(f"找不到token ID {token_id} 的反向映射")
                    original_ids.append(token_id)  # 保持原样
            
            return self.tokenizer.decode(original_ids, skip_special_tokens=True)
        except Exception as e:
            logger.error(f"解码失败: {e}")
            return f"<decode_error: {token_ids}>"


def main():
    parser = argparse.ArgumentParser(description='基于Tokenizer的数据集脱敏转换工具')
    parser.add_argument('--input', '-i', 
                       default="/mnt/shared/data/replay-logs-origin.log",
                       help='输入日志文件路径')
    parser.add_argument('--output', '-o', 
                       default="replay-logs-desensitize.jsonl",
                       help='输出脱敏JSONL文件路径')
    parser.add_argument('--mapping', '-m',
                       default="token_id_mappings.json",
                       help='保存token ID映射关系的文件路径')
    parser.add_argument('--model-name', type=str, 
                       default="Nitral-AI/Captain-Eris_Violet-V0.420-12B",
                       help='用于tokenizer的模型名称')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='显示详细日志')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        return
    
    # 创建脱敏处理器
    desensitizer = TokenizerBasedDesensitizer(model_name=args.model_name)
    
    try:
        # 处理文件
        desensitizer.process_file(args.input, args.output)
        
        # 保存映射关系
        desensitizer.save_mappings(args.mapping)
        
        logger.info("脱敏处理完成!")
        logger.info(f"脱敏数据集: {args.output}")
        logger.info(f"映射文件: {args.mapping}")
        
    except Exception as e:
        logger.error(f"处理过程中出现错误: {e}")
        raise


if __name__ == "__main__":
    main() 