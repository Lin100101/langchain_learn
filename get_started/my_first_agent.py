# 前期准备工作

"""安装langchain包"""
from langchain_openai import ChatOpenAI
from openai import base_url, api_key

# pip install -U langchain
# Requires Python 3.10+



# Installing the OpenAI integration
# pip install -U langchain-openai
#
# Installing the Anthropic integration
# pip install -U langchain-anthropic

"""langchain的官方文档中推荐使用Claude账户，这里我使用的是AiHubMix中转站"""


"""在终端中设置ANTHROPIC_API_KEY环境变量，因为我使用的是windows系统下的pycharm，pycharm默认是使用powershell"""
# $env:ANTHROPIC_API_KEY = "sk-ant-你的实际密钥"
# $env:ANTHROPIC_BASE_URL = "https://你的代理地址"
"""不过我这里是将这些配置，写在了项目中的 .env 文件中"""

import os
from dotenv import load_dotenv # 用于加载 .env 文件加载环境变量
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 自动从 .env 文件加载环境变量
load_dotenv()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"{city}一直很晴朗!"

# 1. 初始化 AIHubMix 模型
# 注意：这里没有显式指定 base_url 和 api_key
"""
0.7（平衡点）：AI 是个“正常的聊天伙伴”。它大部分时候说人话，
但偶尔会给你一点不一样的表达，不会像机器人那样死板，也不会疯言疯语
数值在 0~1之间，数值越大，ai越”感性“，数值越小，ai越”理性“，0.7是一个默认值
这里你可以写，也可以不写，一般来说openai的默认设置是1，其他有些中转站默认是0.7，
建议自己写上
"""
llm = ChatOpenAI(
    model="coding-glm-5-free",  # AIHubMix 想要使用的具体模型名称
    base_url= os.getenv("OPENAI_API_BASE"), # 从环境变量中读取
    api_key=os.getenv("OPENAI_API_KEY"),     # 从环境变量中读取
    temperature=0.7,
)

# 2. 创建 Agent
# 将 model 参数改为上面初始化的 llm 对象
agent = create_agent(
    model=llm,
    tools=[get_weather],
    # 你是一位乐于助人的助手。请始终以中文回复我。
    system_prompt="You are a helpful assistant.Always reply to me in Chinese.",
)

# 3. 运行 Agent
# 注意：invoke 的输入格式必须是 {"messages": [...]}
result = agent.invoke(
    {"messages": [HumanMessage(content="北京的天气怎么样？")]}
)

print(result)