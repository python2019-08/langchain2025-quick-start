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