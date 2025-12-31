import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

# 加载环境变量（确保.env文件中配置了API_KEY）
load_dotenv()
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://api.deepseek.com"

# 初始化LLM模型
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",
    temperature=0.3  # 降低随机性，保证输出稳定
)

# 1. 定义摘要生成提示词（用于压缩对话历史）
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是对话摘要助手，需简洁总结以下对话的核心信息（包含用户身份、偏好、关键问题等），不超过50字。"),
    ("human", "对话历史：{chat_history_text}\n请生成摘要：")
])

# 2. 构建摘要生成链（输入完整历史文本，输出摘要）
summary_chain = summary_prompt | llm

# 3. 定义对话记忆提示词（注入摘要而非完整历史）
summary_memory_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是友好的对话助手，需基于对话摘要回答用户问题，摘要包含核心上下文信息。"),
    ("system", "对话摘要：{chat_summary}"),  # 注入摘要
    ("human", "{user_input}")
])

# 4. 构建基础对话链（提示词 + LLM）
summary_base_chain = (
    RunnablePassthrough.assign(
        chat_summary=lambda x: summary_chain.invoke(
            {
                "chat_history_text": "\n".join(
                    [f"{msg.type}: {msg.content}" for msg in x["chat_history"]]
                )
            }
        ).content
    )
    | summary_memory_prompt
    | llm
)

# 5. 会话历史存储（保存完整历史用于生成摘要）
summary_memory_store = {}

# 6. 定义会话历史获取函数
def get_summary_memory_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in summary_memory_store:
        summary_memory_store[session_id] = InMemoryChatMessageHistory()
    return summary_memory_store[session_id]

# 7. 构建带摘要记忆的对话链
summary_memory_chain = RunnableWithMessageHistory(
    runnable=summary_base_chain,
    get_session_history=get_summary_memory_history,
    input_messages_key="user_input",
    history_messages_key="chat_history"  # 传入完整历史用于生成摘要
)

# 测试多轮对话（session_id=user_003）
config = {"configurable": {"session_id": "user_003"}}

# 多轮对话输入
inputs = [
    "我叫小李，是一名产品经理",
    "我负责一款电商APP的迭代",
    "最近在优化用户下单流程",
    "遇到了用户流失率高的问题",
    "你能给我一些优化建议吗？"
]

for i, user_input in enumerate(inputs, 1):
    response = summary_memory_chain.invoke({"user_input": user_input}, config=config)
    print(f"\n第{i}轮 - 助手回复：", response.content)

# 查看完整历史与最终摘要
history = get_summary_memory_history("user_003")
print("\n摘要记忆的完整对话历史：")
for msg in history.messages:
    print(f"{msg.type}: {msg.content}")

# 单独生成最终摘要验证
final_summary = summary_chain.invoke({
    "chat_history_text": "\n".join([f"{msg.type}: {msg.content}" for msg in history.messages])
}).content
print(f"\n最终对话摘要：{final_summary}")
# 输出示例：摘要：小李，产品经理，负责电商APP迭代，优化下单流程时遇用户流失率高问题，寻求建议。