#!/usr/bin/env python3
"""
测试修改后的脱敏工具JSONL输出功能
"""

import json
import tempfile
import os
import sys
import subprocess

def create_sample_log_file():
    """创建示例日志文件"""
    sample_lines = [
        '2024-01-01T10:00:01Z {"message": "[Log chat request] {\\"conversationId\\": \\"user_12345_session_abc\\", \\"body\\": {\\"prompt\\": [{\\"role\\": \\"user\\", \\"content\\": \\"你好，请帮我写一个Python程序\\"}]}}"}',
        '2024-01-01T10:00:02Z {"message": "[Log chat request] {\\"conversationId\\": \\"user_67890_session_def\\", \\"body\\": {\\"prompt\\": [{\\"role\\": \\"user\\", \\"content\\": \\"Hello, can you help me with machine learning?\\"}]}}"}',
        '2024-01-01T10:00:03Z {"message": "[Log chat request] {\\"conversationId\\": \\"user_12345_session_abc\\", \\"body\\": {\\"prompt\\": [{\\"role\\": \\"user\\", \\"content\\": \\"谢谢，这个函数看起来不错\\"}]}}"}'
    ]
    
    return sample_lines

def test_desensitize_jsonl():
    """测试脱敏工具的JSONL输出功能"""
    print("🧪 开始测试脱敏工具JSONL输出功能...")
    
    # 创建示例日志文件
    sample_lines = create_sample_log_file()
    print(f"✅ 创建了 {len(sample_lines)} 行示例日志")
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False, encoding='utf-8') as temp_log:
        for line in sample_lines:
            temp_log.write(line + '\n')
        temp_log_path = temp_log.name
    
    temp_jsonl_path = temp_log_path.replace('.log', '_desensitized.jsonl')
    temp_mapping_path = temp_log_path.replace('.log', '_mappings.json')
    
    try:
        print(f"📝 临时日志文件: {temp_log_path}")
        print(f"📝 输出JSONL文件: {temp_jsonl_path}")
        print(f"📝 映射文件: {temp_mapping_path}")
        
        # 运行脱敏工具（使用简单的tokenizer避免网络问题）
        print("🔄 运行脱敏工具...")
        
        # 检查脚本是否存在
        script_path = "replay_logs_desensitize/desensitize_dataset_tokenizer.py"
        if not os.path.exists(script_path):
            print(f"❌ 脚本文件不存在: {script_path}")
            return
        
        # 运行脚本（使用gpt2作为备用tokenizer）
        cmd = [
            sys.executable, script_path,
            "--input", temp_log_path,
            "--output", temp_jsonl_path,
            "--mapping", temp_mapping_path,
            "--model-name", "gpt2",  # 使用较小的模型避免网络问题
            "--verbose"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                print(f"❌ 脱敏工具运行失败:")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return
            else:
                print("✅ 脱敏工具运行成功")
        except subprocess.TimeoutExpired:
            print("❌ 脱敏工具运行超时")
            return
        except Exception as e:
            print(f"❌ 运行脱敏工具时出错: {e}")
            return
        
        # 验证输出文件
        if not os.path.exists(temp_jsonl_path):
            print("❌ JSONL文件未生成")
            return
        
        print("🔍 验证JSONL文件格式...")
        
        # 读取JSONL文件并验证格式
        jsonl_items = []
        with open(temp_jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    jsonl_items.append(item)
                    
                    # 验证必要字段
                    if 'timestamp' not in item:
                        print(f"❌ 第{line_num}行缺少timestamp字段")
                        return
                    if 'conversation_id' not in item:
                        print(f"❌ 第{line_num}行缺少conversation_id字段")
                        return
                    if 'messages' not in item:
                        print(f"❌ 第{line_num}行缺少messages字段")
                        return
                    
                    # 验证没有metadata字段
                    if 'metadata' in item:
                        print(f"❌ 第{line_num}行包含不应存在的metadata字段")
                        return
                    
                except json.JSONDecodeError as e:
                    print(f"❌ 第{line_num}行JSON解析失败: {e}")
                    return
        
        print(f"✅ JSONL格式验证通过，共 {len(jsonl_items)} 行")
        
        # 显示JSONL文件内容示例
        print("\n📋 JSONL文件内容示例:")
        for i, item in enumerate(jsonl_items[:2], 1):
            print(f"第{i}行:")
            print(f"  - timestamp: {item['timestamp']}")
            print(f"  - conversation_id: {item['conversation_id']}")
            print(f"  - messages数量: {len(item['messages'])}")
            if item['messages']:
                first_msg = item['messages'][0]
                if 'content' in first_msg and isinstance(first_msg['content'], list):
                    print(f"  - 第一个消息token数量: {len(first_msg['content'])}")
                    print(f"  - 前5个token IDs: {first_msg['content'][:5]}")
        
        # 验证映射文件
        if os.path.exists(temp_mapping_path):
            print("\n🔍 验证映射文件...")
            with open(temp_mapping_path, 'r', encoding='utf-8') as f:
                mappings = json.load(f)
            
            required_fields = ['model_name', 'tokenizer_vocab_size', 'shuffle_table', 'conversation_id_mapping', 'statistics']
            for field in required_fields:
                if field not in mappings:
                    print(f"❌ 映射文件缺少{field}字段")
                    return
            
            print("✅ 映射文件格式正确")
            print(f"  - 模型名称: {mappings['model_name']}")
            print(f"  - 词汇表大小: {mappings['tokenizer_vocab_size']}")
            print(f"  - 置换表大小: {mappings['statistics']['shuffle_table_size']}")
            print(f"  - 处理的会话数: {mappings['statistics']['processed_conversations']}")
        
        # 检查文件大小
        jsonl_size = os.path.getsize(temp_jsonl_path)
        print(f"\n📊 文件统计:")
        print(f"  - JSONL文件大小: {jsonl_size} 字节")
        print(f"  - 平均每行大小: {jsonl_size / len(jsonl_items):.1f} 字节")
        
        print("\n✅ 所有测试通过! 脱敏工具JSONL输出功能正常")
        
    finally:
        # 清理临时文件
        for path in [temp_log_path, temp_jsonl_path, temp_mapping_path]:
            if os.path.exists(path):
                os.unlink(path)
        print("🗑️ 临时文件清理完成")

if __name__ == "__main__":
    test_desensitize_jsonl() 