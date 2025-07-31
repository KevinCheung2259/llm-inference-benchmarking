# 基于置换表的数据集脱敏转换工具

这个工具使用指定模型的tokenizer对文本进行分词，然后通过随机生成的置换表（shuffle table）对token IDs进行重排，保留原始数据的token分布特征，输出为JSON格式。

## 核心特性

1. **置换表脱敏**: 生成与原词表等长的随机置换表，确保一对一映射
2. **Token分布保持**: 保持原始token的统计分布特征
3. **完整性保证**: 
   - 每个原始token ID都有唯一的映射
   - 没有token ID重复或丢失
   - 保持词汇表的完整性
4. **JSON格式输出**: 便于后续处理和分析
5. **可复现性**: 使用固定种子确保结果可复现

## 使用方法

### 基本用法

```bash
python desensitize_dataset_tokenizer.py \
    --input /mnt/shared/data/replay-logs-origin.log \
    --output replay-logs-tokenized.json \
    --model-name "Nitral-AI/Captain-Eris_Violet-V0.420-12B"
```

### 完整参数

```bash
python desensitize_dataset_tokenizer.py \
    --input /mnt/shared/data/replay-logs-origin.log \
    --output replay-logs-tokenized.json \
    --mapping token_id_shuffle_table.json \
    --model-name "Nitral-AI/Captain-Eris_Violet-V0.420-12B" \
    --verbose
```

### 参数说明

- `--input, -i`: 输入的原始日志文件路径
- `--output, -o`: 输出的脱敏JSON文件路径 (默认: replay-logs-tokenized.json)
- `--mapping, -m`: 保存置换表的JSON文件路径 (默认: token_id_mappings.json)
- `--model-name`: 用于tokenizer的模型名称 (默认: "Nitral-AI/Captain-Eris_Violet-V0.420-12B")
- `--verbose, -v`: 显示详细处理日志

## 输出格式

### 1. 脱敏数据集 (JSON格式)

```json
[
  {
    "timestamp": "2024-01-01T10:00:01Z",
    "conversation_id": "conv_a1b2c3d4_e5f6g7h8",
    "messages": [
      {
        "role": "user",
        "content": [1234, 5678, 9012, 3456]  // 脱敏后的token IDs
      },
      {
        "role": "assistant", 
        "content": [7890, 1234, 5678]
      }
    ],
    "metadata": {
      "model_name": "Nitral-AI/Captain-Eris_Violet-V0.420-12B",
      "tokenizer_vocab_size": 50257,
      "original_fields": {}
    }
  }
]
```

### 2. 置换表文件 (token_id_mappings.json)

```json
{
  "model_name": "Nitral-AI/Captain-Eris_Violet-V0.420-12B",
  "tokenizer_vocab_size": 50257,
  "shuffle_table": {
    "0": 12345,
    "1": 67890,
    "2": 54321,
    ...
  },
  "conversation_id_mapping": {
    "original_conv_id": "conv_hash1_hash2"
  },
  "statistics": {
    "processed_lines": 10000,
    "skipped_lines": 50,
    "processed_conversations": 1200,
    "shuffle_table_size": 50257,
    "conversation_id_mappings": 1200
  }
}
```

## 脱敏策略

### 1. 置换表生成

```python
# 生成与原词表等长的置换表
original_ids = [0, 1, 2, ..., vocab_size-1]
shuffled_ids = [0, 1, 2, ..., vocab_size-1] 

# 随机打乱
random.seed(42)  # 固定种子确保可复现
random.shuffle(shuffled_ids)

# 创建一对一映射
shuffle_table = {original_ids[i]: shuffled_ids[i] for i in range(vocab_size)}
```

### 2. 文本处理流程

```python
# 1. 原始文本
text = "你好，请帮我写一个Python程序"

# 2. Tokenizer分词
token_ids = tokenizer.encode(text)  # [123, 456, 789, ...]

# 3. 置换表映射
desensitized_ids = [shuffle_table[tid] for tid in token_ids]  # [987, 654, 321, ...]

# 4. 保存到JSON
{
  "role": "user",
  "content": desensitized_ids
}
```

### 3. 特性保证

- **一对一映射**: 每个原始token ID唯一对应一个脱敏token ID
- **分布保持**: 高频token仍然是高频token，低频token仍然是低频token
- **语义去除**: 完全去除了原始文本的语义信息
- **结构保持**: 保持了对话结构、角色信息和时间戳

## 验证脱敏数据

可以通过以下方式验证脱敏数据的正确性：

```python
# 1. 检查token ID范围
assert all(0 <= tid < vocab_size for message in data for tid in message['content'])

# 2. 检查置换表完整性
assert len(shuffle_table) == vocab_size
assert len(set(shuffle_table.values())) == vocab_size  # 一对一映射

# 3. 检查数据格式
assert all('timestamp' in item for item in data)
assert all('conversation_id' in item for item in data)
assert all('messages' in item for item in data)
```

## 使用脱敏数据进行测试

### 1. 直接使用Token IDs

脱敏数据可以直接用于测试tokenizer性能、模型推理速度等，无需解码：

```python
# 构造测试请求
for item in desensitized_data:
    for message in item['messages']:
        if message['role'] == 'user':
            token_ids = message['content']
            # 直接使用token_ids进行模型推理测试
            test_model_inference(token_ids)
```

### 2. 模拟重放测试

```python
# 转换为模型API格式进行重放测试
requests = []
for item in desensitized_data:
    request = {
        "model": "your-model-name",
        "messages": item['messages'],  # 直接使用token IDs
        "max_tokens": 200,
        "timestamp": item['timestamp']
    }
    requests.append(request)

# 按时间戳重放
for request in sorted(requests, key=lambda x: x['timestamp']):
    send_test_request(request)
```

## 注意事项

1. **模型依赖**: 需要能够访问指定的tokenizer模型
2. **内存使用**: 大词汇表的置换表会占用一定内存
3. **保密性**: 置换表文件应当保密存储
4. **一致性**: 使用相同种子确保多次运行结果一致
5. **兼容性**: 生成的token IDs必须在目标模型的词汇表范围内

## 测试示例

运行测试脚本验证功能：

```bash
# 使用简单tokenizer测试
python test_shuffle_tokenizer.py

# 使用实际模型测试（需要网络访问）
python test_tokenizer_desensitize.py
```

## 性能特点

- **处理速度**: 主要受tokenizer速度限制
- **内存使用**: O(vocab_size) 用于存储置换表
- **文件大小**: JSON格式比原始日志略大，但结构清晰
- **可扩展性**: 支持大规模数据集处理

## 与原始脱敏工具对比

| 特性 | 哈希映射版本 | 置换表版本 |
|------|-------------|------------|
| 映射方式 | 哈希函数 | 随机置换表 |
| 一对一保证 | ❌ | ✅ |
| 分布保持 | 部分 | 完全 |
| 词汇表完整性 | ❌ | ✅ |
| 内存使用 | 低 | 中等 |
| 可复现性 | ✅ | ✅ |
| 输出格式 | 文本 | Token IDs | 