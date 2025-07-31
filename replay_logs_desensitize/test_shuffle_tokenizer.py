#!/usr/bin/env python3
"""
测试置换表脱敏功能的脚本
使用简单的字符tokenizer避免网络依赖
"""

import json
import tempfile
import os
import random
from datetime import datetime

class SimpleTokenizer:
    """简单的字符级tokenizer，用于测试"""
    
    def __init__(self, vocab_size=1000):
        # 创建简单的词汇表：常用字符 + 数字 + 英文字母 + 中文字符
        vocab = []
        
        # 添加特殊token
        special_tokens = ['<pad>', '<unk>', '<start>', '<end>']
        vocab.extend(special_tokens)
        
        # 添加基本字符
        for i in range(32, 127):  # 基本ASCII字符
            vocab.append(chr(i))
        
        # 添加一些常用中文字符
        common_chinese = "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严"
        for char in common_chinese:
            if char not in vocab:
                vocab.append(char)
        
        # 补充到指定大小
        while len(vocab) < vocab_size:
            vocab.append(f"<unk_{len(vocab)}>")
        
        self.vocab = vocab[:vocab_size]
        self.vocab_size = len(self.vocab)
        self.char_to_id = {char: i for i, char in enumerate(self.vocab)}
        self.id_to_char = {i: char for i, char in enumerate(self.vocab)}
    
    def encode(self, text, add_special_tokens=False):
        """将文本编码为token IDs"""
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.char_to_id.get('<start>', 2))
        
        for char in text:
            token_id = self.char_to_id.get(char, self.char_to_id.get('<unk>', 1))
            token_ids.append(token_id)
        
        if add_special_tokens:
            token_ids.append(self.char_to_id.get('<end>', 3))
        
        return token_ids
    
    def decode(self, token_ids, skip_special_tokens=False):
        """将token IDs解码为文本"""
        chars = []
        special_tokens = {'<pad>', '<unk>', '<start>', '<end>'}
        
        for token_id in token_ids:
            if token_id < self.vocab_size:
                char = self.id_to_char[token_id]
                if skip_special_tokens and char in special_tokens:
                    continue
                # 跳过生成的unk token
                if char.startswith('<unk_'):
                    continue
                chars.append(char)
        
        return ''.join(chars)
    
    def __len__(self):
        return self.vocab_size


class SimpleTokenizerBasedDesensitizer:
    """使用简单tokenizer的脱敏处理器"""
    
    def __init__(self, vocab_size=1000):
        self.tokenizer = SimpleTokenizer(vocab_size)
        self.vocab_size = len(self.tokenizer)
        
        # 映射关系
        self.token_id_mapping = {}
        self.conversation_id_mapping = {}
        
        # 统计信息
        self.processed_lines = 0
        self.skipped_lines = 0
        self.processed_conversations = set()
        
        # 创建token ID的置换表
        self._create_shuffle_table()
    
    def _create_shuffle_table(self):
        """创建token ID置换表"""
        print("正在创建token ID置换表...")
        
        # 生成与原词表等长的置换表
        original_ids = list(range(self.vocab_size))
        shuffled_ids = list(range(self.vocab_size))
        
        # 使用固定种子确保可复现性
        random.seed(42)
        random.shuffle(shuffled_ids)
        
        # 创建一对一的映射关系（置换表）
        for i, original_id in enumerate(original_ids):
            self.token_id_mapping[original_id] = shuffled_ids[i]
        
        print(f"创建了 {len(self.token_id_mapping)} 个token ID映射的置换表")
    
    def _desensitize_text(self, text):
        """对文本进行tokenizer分词并返回脱敏后的token IDs"""
        if not text:
            return []
        
        # 使用tokenizer进行分词
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        
        # 使用置换表对token IDs进行脱敏映射
        desensitized_ids = []
        for token_id in token_ids:
            if token_id < self.vocab_size:
                desensitized_ids.append(self.token_id_mapping[token_id])
            else:
                print(f"警告: Token ID {token_id} 超出词汇表范围 {self.vocab_size}")
                desensitized_ids.append(token_id % self.vocab_size)
        
        return desensitized_ids
    
    def _desensitize_conversation_id(self, conversation_id):
        """对会话ID进行脱敏"""
        if conversation_id in self.conversation_id_mapping:
            return self.conversation_id_mapping[conversation_id]
        
        # 生成新的会话ID
        import hashlib
        hash_obj = hashlib.md5(conversation_id.encode('utf-8'))
        hex_hash = hash_obj.hexdigest()
        new_id = f"conv_{hex_hash[:8]}_{hex_hash[8:16]}"
        self.conversation_id_mapping[conversation_id] = new_id
        
        return new_id
    
    def decode_tokens(self, token_ids):
        """将脱敏后的token IDs解码为文本"""
        # 创建反向置换表
        if not hasattr(self, '_reverse_mapping'):
            self._reverse_mapping = {v: k for k, v in self.token_id_mapping.items()}
        
        # 反向映射token IDs到原始IDs
        original_ids = []
        for token_id in token_ids:
            if token_id in self._reverse_mapping:
                original_ids.append(self._reverse_mapping[token_id])
            else:
                original_ids.append(token_id)
        
        return self.tokenizer.decode(original_ids, skip_special_tokens=True)


def create_sample_log_data():
    """创建示例日志数据"""
    sample_data = [
        {
            "timestamp": "2024-01-01T10:00:01Z",
            "conversationId": "user_12345_session_abc",
            "body": {
                "prompt": [
                    {"role": "user", "content": "你好，请帮我写一个Python程序"},
                    {"role": "assistant", "content": "好的，我可以帮您写Python程序。请告诉我您具体需要什么功能？"},
                    {"role": "user", "content": "我需要一个计算斐波那契数列的函数"}
                ]
            }
        },
        {
            "timestamp": "2024-01-01T10:00:02Z", 
            "conversationId": "user_67890_session_def",
            "body": {
                "prompt": [
                    {"role": "user", "content": "Hello, can you help me with machine learning?"},
                    {"role": "assistant", "content": "Of course! I'd be happy to help you with machine learning."}
                ]
            }
        }
    ]
    
    log_lines = []
    for data in sample_data:
        message = f"[Log chat request] {json.dumps(data, ensure_ascii=False)}"
        log_json = {"message": message}
        log_line = f"{data['timestamp']} {json.dumps(log_json, ensure_ascii=False)}"
        log_lines.append(log_line)
    
    return log_lines


def test_shuffle_tokenizer():
    """测试置换表脱敏功能"""
    print("🧪 开始测试置换表脱敏工具...")
    
    # 创建脱敏处理器
    desensitizer = SimpleTokenizerBasedDesensitizer(vocab_size=500)
    print("✅ 创建脱敏处理器成功")
    print(f"词汇表大小: {desensitizer.vocab_size}")
    
    # 测试文本脱敏
    test_texts = [
        "你好，世界！",
        "Hello, World!",
        "Python编程语言",
        "Machine Learning"
    ]
    
    print("\n🔍 测试文本脱敏:")
    for text in test_texts:
        print(f"\n原始文本: '{text}'")
        
        # 编码
        original_ids = desensitizer.tokenizer.encode(text)
        print(f"原始token IDs: {original_ids}")
        
        # 脱敏
        desensitized_ids = desensitizer._desensitize_text(text)
        print(f"脱敏token IDs: {desensitized_ids}")
        
        # 解码验证
        decoded_text = desensitizer.decode_tokens(desensitized_ids)
        print(f"解码后文本: '{decoded_text}'")
        
        # 验证解码后是否与原文一致
        if decoded_text == text:
            print("✅ 解码验证成功")
        else:
            print("❌ 解码验证失败")
    
    # 显示一些置换表样例
    print("\n🗝️ 置换表样例:")
    mapping_items = list(desensitizer.token_id_mapping.items())[:10]
    for original_id, mapped_id in mapping_items:
        original_char = desensitizer.tokenizer.id_to_char.get(original_id, "?")
        mapped_char = desensitizer.tokenizer.id_to_char.get(mapped_id, "?")
        print(f"  {original_id} ('{original_char}') -> {mapped_id} ('{mapped_char}')")
    
    # 验证置换表的完整性
    print(f"\n📊 置换表统计:")
    print(f"- 映射数量: {len(desensitizer.token_id_mapping)}")
    print(f"- 词汇表大小: {desensitizer.vocab_size}")
    print(f"- 映射覆盖率: {len(desensitizer.token_id_mapping)/desensitizer.vocab_size*100:.1f}%")
    
    # 验证置换表的一对一映射
    mapped_values = list(desensitizer.token_id_mapping.values())
    unique_values = set(mapped_values)
    if len(mapped_values) == len(unique_values):
        print("✅ 置换表是一对一映射")
    else:
        print("❌ 置换表存在重复映射")
    
    # 验证所有token ID都在词汇表范围内
    all_in_range = all(0 <= v < desensitizer.vocab_size for v in mapped_values)
    if all_in_range:
        print("✅ 所有映射值都在词汇表范围内")
    else:
        print("❌ 存在超出词汇表范围的映射值")
    
    print("\n✅ 置换表脱敏工具测试完成!")


if __name__ == "__main__":
    test_shuffle_tokenizer() 