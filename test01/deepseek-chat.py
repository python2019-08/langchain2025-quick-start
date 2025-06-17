import os
import sys

# ------------------------
now_dir = os.getcwd()
sys.path.append(now_dir)

from config.load_key import load_key
apiKey =load_key("deepseek-api-key")
print(apiKey)

# ------------------------
# 制定 OpenAI 的 API_KEY 。
if not os.environ.get("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = load_key("deepseek-api-key")


from langchain_core.messages import HumanMessage,SystemMessage    
from langchain_deepseek import ChatDeepSeek
try:
    # 调用deepseekd的代码，在ChatDeepSeek中就有注释
    llm = ChatDeepSeek(
        model="deepseek-chat" , # deepseek的模型名目前包括 deepseek-chat 和 deepseek-reasonser
    )
    ret=llm.invoke([HumanMessage("你是谁？你能帮我解决什么问题？")])
    print(ret)
except Exception as e:
    if "402" in str(e) or "Insufficient Balance" in str(e):
        print("❌ 错误：DeepSeek API调用失败，账户余额不足。请充值后再试。")
        print("🔗 请访问DeepSeek平台查看账户余额：https://deepseek.com/account")
    else:
        print(f"❌ 其他错误：{e}")    