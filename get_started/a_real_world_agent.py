# 天气预报 agent
"""
构建只能体的六个关键步骤
1. 详细的系统提示词 System Prompts
2. 创建集成外部数据的工具
3. 一致响应的模型配置
4. 结构化输出
5. 对话记忆(类似于聊天)
6. 创建并运行
"""



from dotenv import load_dotenv # 用于加载 .env 文件加载环境变量
# 自动从 .env 文件加载环境变量
load_dotenv()
import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# 1. 详细的系统提示词 System Prompts
SYSTEM_PROMPT = """你是一位擅长用双关语表达的天气预报专家。
您拥有以下两种工具的使用权：

- get_weather_for_location: 用它来获取特定地点的天气情况
- get_user_location: 用这个来获取用户的地理位置

如果用户向你询问天气情况，务必将其所在位置搞清楚。
如果从问题中可以判断出他们指的是他们当前所在的地方，
则使用“获取用户位置”工具来确定他们的位置。"""

# 2. 创建集成外部数据的工具
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime

@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"{city}天气一直很晴朗!"

@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "北京" if user_id == "1" else "SF"

# 3. 配置模型
from langchain.chat_models import init_chat_model

llm = ChatOpenAI(
    model="coding-glm-5-free",
    base_url=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7,
)

# 4. 定义响应格式
from dataclasses import dataclass

# We use a dataclass here, but Pydantic models are also supported.
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None

# 5. 添加到内存
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

# 6. 创建并运行
from langchain.agents.structured_output import ToolStrategy

agent = create_agent(
    model=llm,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)

# `thread_id` is a unique identifier for a given conversation.
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "外面天气怎么样?"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(
#     punny_response="Florida is still having a 'sun-derful' day! The sunshine is playing 'ray-dio' hits all day long! I'd say it's the perfect weather for some 'solar-bration'! If you were hoping for rain, I'm afraid that idea is all 'washed up' - the forecast remains 'clear-ly' brilliant!",
#     weather_conditions="It's always sunny in Florida!"
# )


# Note that we can continue the conversation using the same `thread_id`.
response = agent.invoke(
    {"messages": [{"role": "user", "content": "谢谢你!"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(
#     punny_response="You're 'thund-erfully' welcome! It's always a 'breeze' to help you stay 'current' with the weather. I'm just 'cloud'-ing around waiting to 'shower' you with more forecasts whenever you need them. Have a 'sun-sational' day in the Florida sunshine!",
#     weather_conditions=None
# )