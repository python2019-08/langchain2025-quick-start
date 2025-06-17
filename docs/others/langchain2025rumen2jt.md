# 1.2025 最强LangChain从入门到精通教程
 
刷爆全网！2025 最强LangChain从入门到精通教程！底层原理 + 实战项目全解！存下直接跟着学，少走99%冤枉路！
2025-06-14 13:47:00
https://www.bilibili.com/video/BV1uFM2zEEHk/?spm_id_from=333.337.search-card.all.click&vd_source=4212b105520112daf65694a1e5944e23

```sh
 $ conda activate langch   
 $ conda deactivate
```

# 【基础篇】02.api_key设置

```sh
!pip install -q chromadb
!pip install -q tiktoken
!pip install -qU langchain-openai
!pip install -qU langchain-community
!pip install unstructured
```

```py
import os

# 腾讯混元大模型 --> 使用OpenAI SDK方式接入 
with open("openai_api_key.txt") as f:
    OPENAI_API_KEY= f.read()

os.environ['OPENAI_API_KEY']= OPENAI_API_KEY

#import os
#from Langchain_google_genai import ChatGoogleGenerativeAI
##假设 GOOGLE_API_KEY 已经设置为环境变量
#os.environ["GooGLE_API_KEY"]="你的_gemini_api_密钥"
#gemini_LLm=ChatGoogLeGenerativeAI(modeL="gemini-pro"）#或者其他模型
# 
#import os
#from Langchain_anthropicimportChatAnthropic
##假设 ANTHROPIC_API_KEY 已经设置为环境变量
#os.environ["ANTHROPIC_API_KEY"]="你的_cLaude_api_密钥
#cLaude_LLm=ChatAnthropic(modeL="cLaude-3-opus-20240229"）#或者其他模型

#import os
#from Langchain_deepseek import ChatDeepseek
##假设DEEPSEEK_API_KEY已经设置为环境变量
#os.environ["DEEPSEEK_API_KEY"]="你的_deepseek_api_密钥"
#deepseek_LLm=ChatDeepseek(modeL="deepseek-chat"）#或者其他模型

#  baidu qianfan API key
#       bce-v3/ALTAK-gkYESB8PyvAXslrL47olN/48f891458c251a6a66f0cbbee00075985475ceaf
```