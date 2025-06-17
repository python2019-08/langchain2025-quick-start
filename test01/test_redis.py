import redis
import json

# 连接到 Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# 获取所有键（调试用，生产环境慎用 KEYS 命令）
all_keys = r.keys('*')
print("所有键:", all_keys)

# 获取特定 session_id 的数据
session_id = "chat:test:01JXY6DV3CVV6EQB5FAGJY3G2F"
data = r.get(session_id)

if data:
    try:
        # 尝试解析为 JSON
        parsed_data = json.loads(data)
        print("JSON 数据:", parsed_data)
    except json.JSONDecodeError:
        # 可能是其他格式（如列表、哈希）
        print("原始数据:", data)
        
        # 检查是否为列表
        if r.type(session_id) == b'list':
            list_data = r.lrange(session_id, 0, -1)
            print("列表数据:", [item.decode('utf-8') for item in list_data])
            
        # 检查是否为哈希
        elif r.type(session_id) == b'hash':
            hash_data = r.hgetall(session_id)
            print("哈希数据:", {k.decode('utf-8'): v.decode('utf-8') for k, v in hash_data.items()})
else:
    print(f"未找到 session_id 为 '{session_id}' 的数据。")