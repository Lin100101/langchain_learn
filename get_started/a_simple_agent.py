from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# 加载配置文件 .env
load_dotenv()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"{city}天气一直很晴朗!"

llm = ChatOpenAI(
    model="coding-glm-5-free",
    base_url=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7
)
agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="你是一个得力助手。"
)
# Run the agent
response = agent.invoke(
    {"messages": [{"role": "user", "content": "北京的天气怎么样？"}]}
)

print(response["messages"][-1].content)