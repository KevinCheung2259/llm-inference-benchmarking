import openai
import asyncio

ep_config = {
    "api_key": "YOUR_API_KEY",  # 替换为你的 OpenAI API Key
    "api_base": "http://184.105.217.134:8000/v1"  # 替换为你的 OpenAI API Base URL（如果有）
}

client = openai.AsyncOpenAI(
    base_url=ep_config["api_base"], api_key=ep_config["api_key"]
)

def read_tokens(file_path):
    with open(file_path, 'r') as file:
        tokens = file.read().strip().splitlines()
    return tokens


async def make_request(token, line_num):
    messages = [
        {"role": "user", "content": token + " help me"},
    ]
    try:
        response = await client.chat.completions.create(
            model="Nitral-AI/Captain-Eris_Violet-V0.420-12B",
            messages=messages,
            max_tokens=50,
            temperature=0,
            # Do not set to false. You will get bogus results.
            stream=True,
            stream_options={"include_usage": True},
        )
        print(f"Request for line {line_num} succeeded: {response}")
    except Exception as e:
        print(f"Error for line {line_num} with token '{token}': {e}")


async def main():
    tokens = read_tokens('tokens.txt')
    tasks = []
    for i, token in enumerate(tokens):
        task = asyncio.ensure_future(make_request(token, i + 1)) 
        tasks.append(task)

    await asyncio.gather(*tasks)

# 运行事件循环
if __name__ == "__main__":
    asyncio.run(main())