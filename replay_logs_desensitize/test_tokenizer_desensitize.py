#!/usr/bin/env python3
"""
测试基于Tokenizer的脱敏工具
"""

import json
import tempfile
import os
from datetime import datetime
from desensitize_dataset_tokenizer import TokenizerBasedDesensitizer

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
                    {"role": "assistant", "content": "Of course! I'd be happy to help you with machine learning. What specific topic are you interested in?"}
                ]
            }
        },
        {
            "timestamp": "2024-01-01T10:00:03Z",
            "conversationId": "user_12345_session_abc",  # 重复的会话ID
            "body": {
                "prompt": [
                    {"role": "user", "content": "谢谢，这个函数看起来不错"}
                ]
            }
        }
    ]
    
    log_lines = []
    for data in sample_data:
        # 构造日志行格式
        message = f"[Log chat request] {json.dumps(data, ensure_ascii=False)}"
        log_json = {"message": message}
        log_line = f"{data['timestamp']} {json.dumps(log_json, ensure_ascii=False)}"
        log_lines.append(log_line)
    
    return log_lines

def test_tokenizer_desensitizer():
    """测试基于tokenizer的脱敏功能"""
    print("🧪 开始测试基于Tokenizer的脱敏工具...")
    
    # 创建示例数据
    sample_lines = create_sample_log_data()
    print(f"✅ 创建了 {len(sample_lines)} 行示例数据")
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False, encoding='utf-8') as temp_input:
        for line in sample_lines:
            temp_input.write(line + '\n')
        temp_input_path = temp_input.name
    
    temp_output_path = temp_input_path.replace('.log', '_tokenized.json')
    temp_mapping_path = temp_input_path.replace('.log', '_mappings.json')
    
    try:
        print(f"📝 临时输入文件: {temp_input_path}")
        print(f"📝 临时输出文件: {temp_output_path}")
        
        # 创建脱敏处理器（使用gpt2作为测试tokenizer，避免下载大模型）
        print("🔄 正在初始化tokenizer...")
        desensitizer = TokenizerBasedDesensitizer(model_name="gpt2")
        print("✅ 创建脱敏处理器成功")
        
        # 处理文件
        print("🔄 开始处理文件...")
        desensitizer.process_file(temp_input_path, temp_output_path)
        print("✅ 文件处理完成")
        
        # 保存映射关系
        desensitizer.save_mappings(temp_mapping_path)
        print("✅ 映射关系保存完成")
        
        # 读取并显示结果
        print("\n📊 处理结果:")
        print(f"- 处理行数: {desensitizer.processed_lines}")
        print(f"- 跳过行数: {desensitizer.skipped_lines}")
        print(f"- 会话数: {len(desensitizer.processed_conversations)}")
        print(f"- Token ID映射数: {len(desensitizer.token_id_mapping)}")
        print(f"- Tokenizer词汇表大小: {desensitizer.vocab_size}")
        
        # 读取生成的JSON文件
        with open(temp_output_path, 'r', encoding='utf-8') as f:
            output_data = json.load(f)
        
        print(f"\n📋 生成的JSON数据包含 {len(output_data)} 个条目")
        
        # 显示第一个条目的详细信息
        if output_data:
            first_item = output_data[0]
            print("\n🔍 第一个条目详细信息:")
            print(f"- 时间戳: {first_item['timestamp']}")
            print(f"- 脱敏会话ID: {first_item['conversation_id']}")
            print(f"- 消息数量: {len(first_item['messages'])}")
            
            # 显示第一个消息的token化结果
            if first_item['messages']:
                first_message = first_item['messages'][0]
                print(f"- 第一个消息角色: {first_message.get('role', 'N/A')}")
                if 'content' in first_message:
                    token_ids = first_message['content']
                    print(f"- Token IDs (前10个): {token_ids[:10]}")
                    print(f"- Token IDs 总数: {len(token_ids)}")
                    
                    # 尝试解码token以验证
                    decoded_text = desensitizer.decode_tokens(token_ids)
                    print(f"- 解码后的文本: {decoded_text[:50]}...")
        
        # 显示一些token ID映射样例
        print("\n🗝️ Token ID映射样例:")
        mapping_items = list(desensitizer.token_id_mapping.items())[:10]
        for original_id, mapped_id in mapping_items:
            try:
                original_token = desensitizer.tokenizer.decode([original_id])
                mapped_token = desensitizer.tokenizer.decode([mapped_id])
                print(f"  {original_id} ('{original_token}') -> {mapped_id} ('{mapped_token}')")
            except:
                print(f"  {original_id} -> {mapped_id}")
        
        # 显示会话ID映射
        print("\n🆔 会话ID映射:")
        for original, mapped in desensitizer.conversation_id_mapping.items():
            print(f"  '{original}' -> '{mapped}'")
        
        print("\n✅ 测试完成! 基于Tokenizer的脱敏工具工作正常。")
        
        # 验证JSON格式
        print("\n🔍 验证JSON格式...")
        validate_json_format(temp_output_path)
        
    finally:
        # 清理临时文件
        for path in [temp_input_path, temp_output_path, temp_mapping_path]:
            if os.path.exists(path):
                os.unlink(path)
        print("🗑️ 临时文件清理完成")

def validate_json_format(json_file):
    """验证生成的JSON文件格式"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    assert isinstance(data, list), "顶层应该是列表"
    
    for i, item in enumerate(data):
        assert isinstance(item, dict), f"第{i}个条目应该是字典"
        assert 'timestamp' in item, f"第{i}个条目缺少timestamp字段"
        assert 'conversation_id' in item, f"第{i}个条目缺少conversation_id字段"
        assert 'messages' in item, f"第{i}个条目缺少messages字段"
        assert 'metadata' in item, f"第{i}个条目缺少metadata字段"
        
        # 验证messages格式
        messages = item['messages']
        assert isinstance(messages, list), f"第{i}个条目的messages应该是列表"
        
        for j, msg in enumerate(messages):
            if isinstance(msg, dict):
                if 'content' in msg:
                    content = msg['content']
                    assert isinstance(content, list), f"第{i}个条目第{j}个消息的content应该是token ID列表"
                    assert all(isinstance(token_id, int) for token_id in content), f"第{i}个条目第{j}个消息的content应该包含整数token IDs"
    
    print(f"✅ JSON格式验证通过: {len(data)} 个条目")

if __name__ == "__main__":
    test_tokenizer_desensitizer() 