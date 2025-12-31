# 第三章 LangChain进阶组件实操

## 前言

上一章我们了解了LangChain的“核心”——组件，包括模型调用、提示词模板以及输出解析。

从这一章开始，我们将聚焦LangChain生态中的核心进阶组件，从状态管理、外部行动两个核心维度拆解组件原理，再通过组件组合实践掌握复杂应用的构建方法。

通过本章学习，你将具备构建带记忆、能交互外部系统的智能应用的能力。

Go Go Go ，我们就出发吧！

## 3.1 状态管理层（Memory）：让模型拥有记忆能力

大语言模型（LLM）的原生调用是无状态的，即每次对话都是独立的请求，无法主动记住上下文信息。LangChain的Memory组件正是为解决这一问题而生，它通过结构化的方式存储、管理对话历史，让AI具备“记忆能力”。

### 3.1.1 对话记忆的本质与作用

#### 3.1.1.1 核心本质

其实Memory组件的工作逻辑特别好理解，就像我们聊天时记笔记一样：

每次你和AI对话后，它会把“你说的话”和“它的回复”整理好存起来（这一步叫“存储”）；等你下一次提问时，它会先把之前存的笔记拿出来，和你新的问题拼在一起再交给LLM（这一步叫“提取”）。这样LLM就能看到完整的对话上下文，自然就能记住你之前说的内容了。

从技术实现上，Memory组件通过两个核心动作完成工作：

- 存储（Save）：将每一轮的用户输入（HumanMessage）和AI输出（AIMessage）保存到指定存储介质（内存、数据库等）；

- 提取（Load）：新一轮对话时，从存储介质中提取历史对话，注入到Prompt中供LLM参考。

#### 3.1.1.2 核心作用

记忆功能看似简单，但能帮我们解决很多实际问题：

- 避免重复提问：比如你说过“我叫小明”，之后不用再重复，AI也能叫出你的名字；
- 支撑复杂任务：比如你让AI“先梳理我的需求，再生成方案”，它能记住中间的需求梳理结果，不会中途失忆；
- 简化交互：不用每次提问都把前因后果说一遍，比如问“这个组件怎么用？”，AI知道你说的是之前聊的Memory组件。

### 3.1.2 三种基础Memory组件实操

LangChain提供了多种Memory实现，适用于不同场景。我们重点学最常用的三种——全量记忆、窗口记忆、摘要记忆。它们各有适用场景，学会了就能应对大部分需求。

需要注意的是，LangChain 0.2.x及以上版本推荐使用LCEL（LangChain Execution Logic）架构，通过 `RunnableWithMessageHistory` 结合 `BaseChatMessageHistory` 抽象类实现对话记忆管理，替代后续不支持的 `ConversationChain`。

本教程将基于该架构分别实现三种核心记忆模式：

- **全量记忆**：完整保存所有对话历史，适用于短对话场景
- **窗口记忆**：仅保留最近N轮对话，控制Token消耗
- **摘要记忆**：通过LLM生成对话摘要替代完整历史，平衡上下文连贯性与效率

核心优势：支持多会话隔离（通过session_id）、自动管理历史消息的注入与保存、适配现代LLM模型的会话交互逻辑。

【前置准备】所有案例需先完成环境配置：

```python
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
```

#### 3.1.2.1 全量记忆

使用`InMemoryChatMessageHistory` 存储完整对话历史，每次调用时自动注入所有历史消息到提示词中，适用于对话轮数少、需要完整上下文的场景。

```python
# 1. 定义提示词模板（包含历史消息占位符）
full_memory_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是友好的对话助手，需基于完整的历史对话回答用户问题。"),
    MessagesPlaceholder(variable_name="chat_history"),  # 历史消息占位符
    ("human", "{user_input}")  # 用户当前输入
])

# 2. 构建基础链（提示词 + LLM）
base_chain = full_memory_prompt | llm

# 3. 会话历史存储（内存模式，生产环境可替换为数据库存储）
full_memory_store = {}

# 4. 定义会话历史获取函数（核心：返回完整历史）
def get_full_memory_history(session_id: str) -> BaseChatMessageHistory:
    """根据session_id获取会话历史，不存在则创建新的历史记录"""
    if session_id not in full_memory_store:
        full_memory_store[session_id] = InMemoryChatMessageHistory()
    return full_memory_store[session_id]

# 5. 构建带全量记忆的对话链
full_memory_chain = RunnableWithMessageHistory(
    runnable=base_chain,
    get_session_history=get_full_memory_history,
    input_messages_key="user_input",  # 输入中用户问题的键名
    history_messages_key="chat_history"  # 传入提示词的历史消息键名
)
```

**ChatPromptTemplate.from_messages** 创建了一个对话提示模板，它就像是给AI设定了一个“剧本”，规定了对话的结构和角色

**MessagesPlaceholder(variable_name="history")**这是一个历史消息占位符。这是实现对话记忆（记忆功能）的关键。它不在模板中写死任何内容，而是在程序运行时，动态地将之前的对话记录（比如用户之前问了什么，AI回答了什么）插入到这个位置。这样，AI在回答新问题时就能参考上下文，实现连贯的多轮对话

`base_chain = full_memory_prompt | llm`

这行代码使用管道操作符 `|`将两个组件连接起来，形成了一个简单的处理链，其含义是**将前一个组件的输出，作为后一个组件的输入**

- **执行 `prompt`**：`prompt`组件接收一个包含 `input`（用户新问题）和 `history`（历史对话）的变量字典，然后根据模板生成一个结构化的消息列表。

- **管道传递 (`|`)**：这个生成好的消息列表被自动传递给 `llm`（大型语言模型，如 GPT-4）。

- **执行 `llm`**：`llm`组件根据收到的消息列表，生成一段连贯且符合上下文的回答。

测试验证

```python
# 测试多轮对话（指定session_id=user_001，隔离不同用户）
config = {"configurable": {"session_id": "user_001"}}

# 第一轮对话
response1 = full_memory_chain.invoke({"user_input": "我叫小明，喜欢编程"}, config=config)
print("助手回复1：", response1.content)
# 输出示例：你好小明！编程是一项很有创造力的技能，你平时常用什么编程语言呢？

# 第二轮对话（验证记忆：询问历史信息）
response2 = full_memory_chain.invoke({"user_input": "我刚才说我喜欢什么？"}, config=config)
print("助手回复2：", response2.content)
# 输出示例：你刚才说你喜欢编程呀～

# 查看完整历史记录
print("\n全量记忆的对话历史：")
for msg in get_full_memory_history("user_001").messages:
    print(f"{msg.type}: {msg.content}")
```

运行结果

```
助手回复1： 你好小明！很高兴认识你！编程是个非常棒的爱好，能创造、解决问题，还能实现各种有趣的想法。你主要对哪种编程语言或领域感兴趣呢？比如网页开发、数据分析、游戏设计，还是其他方向？ 😊
助手回复2： 你刚才提到你喜欢编程！需要我推荐一些学习资源、项目灵感，或者聊聊编程相关的话题吗？ 😄
助手回复3： 你刚才告诉我，你的名字是**小明**！需要我帮你记录或规划与编程相关的学习目标吗？ 😊

全量记忆的对话历史：
human: 我叫小明，喜欢编程
ai: 你好小明！很高兴认识你！编程是个非常棒的爱好，能创造、解决问题，还能实现各种有趣的想法。你主要对哪种编程语言或领域感兴趣呢？比如网页开发、数据分析、游戏设计，还是其他方向？  😊
human: 我刚才说我喜欢什么？
ai: 你刚才提到你喜欢编程！需要我推荐一些学习资源、项目灵感，或者聊聊编程相关的话题吗？ 😄
human: 我叫什么名字
ai: 你刚才告诉我，你的名字是**小明**！需要我帮你记录或规划与编程相关的学习目标吗？ 😊
```

通过运行结果你发现了什么？

#### 3.1.2.2 窗口记忆

全量记忆适合短对话，那长对话怎么办？

这时候就需要“窗口记忆”——它只保留最近的N轮对话（N用k参数控制），早期的对话会自动丢弃。这样能有效控制文字量，适合客服、长期陪伴等长对话场景。

```python
# 1. 定义提示词模板（与全量记忆通用，可复用）
window_memory_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是友好的对话助手，需基于最近的对话历史回答用户问题。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])

# 2. 构建基础链
window_base_chain = window_memory_prompt | llm

# 3. 会话历史存储
window_memory_store = {}
WINDOW_SIZE = 2  # 保留最近2轮对话（即最近4条消息：用户-助手-用户-助手）

# 4. 定义带窗口限制的会话历史获取函数
def get_window_memory_history(session_id: str) -> BaseChatMessageHistory:
    """获取会话历史，仅保留最近WINDOW_SIZE轮对话"""
    if session_id not in window_memory_store:
        window_memory_store[session_id] = InMemoryChatMessageHistory()
    
    # 获取完整历史，截取最近WINDOW_SIZE轮（每轮2条消息）
    history = window_memory_store[session_id]
    if len(history.messages) > 2 * WINDOW_SIZE:
        # 截取后WINDOW_SIZE轮消息（保留最新的）
        history.messages = history.messages[-2 * WINDOW_SIZE:]
    return history

# 5. 构建带窗口记忆的对话链
window_memory_chain = RunnableWithMessageHistory(
    runnable=window_base_chain,
    get_session_history=get_window_memory_history,
    input_messages_key="user_input",
    history_messages_key="chat_history"
)
```

测试验证

```python
# 测试多轮对话（session_id=user_002，与全量记忆会话隔离）
config = {"configurable": {"session_id": "user_002"}}

# 模拟5轮对话，验证窗口记忆的截断效果
inputs = [
    "我叫小红",
    "我喜欢画画",
    "我来自上海",
    "我是一名学生",
    "我刚才说我来自哪里？"  # 第5轮：询问第3轮的信息，验证窗口截断
]

for i, user_input in enumerate(inputs, 1):
    response = window_memory_chain.invoke({"user_input": user_input}, config=config)
    print(f"\n第{i}轮 - 助手回复：", response.content)

# 查看窗口记忆的最终历史（仅保留最近2轮）
print("\n窗口记忆的最终对话历史（最近2轮）：")
for msg in get_window_memory_history("user_002").messages:
    print(f"{msg.type}: {msg.content}")
```

运行结果

```
第1轮 - 助手回复： 你好小红！很高兴认识你！有什么我可以帮你的吗？

第2轮 - 助手回复： 画画是很棒的爱好呢！你通常喜欢画什么类型的作品？比如风景、人物，还是抽象画？

第3轮 - 助手回复： 上海是个充满艺术气息的城市呢！那里有很多美术馆和创意园区，比如西岸艺术中心、M50创意园，说不定能给你的创作带来灵感哦～

第4轮 - 助手回复： 学生时期能有时间坚持爱好真不容易呢！你是通过学校社团、课外班自学，还是纯粹当作放松的方式呢？

第5轮 - 助手回复： 你刚才提到你来自上海～需要我帮你推荐些适合学生参观的艺术展览或创意市集吗？(๑•̀ㅂ•́)و✧

窗口记忆的最终对话历史（最近2轮）：
human: 我是一名学生
ai: 学生时期能有时间坚持爱好真不容易呢！你是通过学校社团、课外班自学，还是纯粹当作放松的方式呢？
human: 我刚才说我来自哪里？
ai: 你刚才提到你来自上海～需要我帮你推荐些适合学生参观的艺术展览或创意市集吗？(๑•̀ㅂ•́)و✧
```

通过运行结果你发现了什么，模型只保留了2轮的记忆。

#### 3.1.2.3 摘要记忆

如果需要超长时间的对话（比如几小时的咨询），即使是窗口记忆也可能不够用。这时候就需要“摘要记忆”——它不保存对话原文，而是用LLM把历史对话总结成一段简洁的摘要。既能保留核心信息，又能最大程度节省文字量，缺点是可能会丢失一些细节（比如具体的数字、名字）。

```python
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
```

测试验证

```python
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
```

运行结果

```
第1轮 - 助手回复： 你好小李！很高兴认识你！作为产品经理，你的工作一定充满挑战和创意吧？如果需要讨论产品设计、用户需求或任何相关话题，我随时可以帮忙哦！ 😊

第2轮 - 助手回复： 很高兴能为您提供帮助！作为产品经理，您对电商APP的迭代有什么具体方向或问题需要探讨吗？比如用户增长、功能优化、体验提升，或是数据驱动决策等方面？

第3轮 - 助手回复： 好的，小李。优化下单流程是提升转化率和用户体验的关键。基于我们之前的讨论，这里有几个核心方向和具体建议供你参考：
...省略

第4轮 - 助手回复： 根据对话摘要，您正在优化电商APP的下单流程。针对用户流失率高的问题，可以结合AI之前的建议，从以下几个方向入手：

...省略

摘要记忆的完整对话历史：
human: 我叫小李，是一名产品经理
ai: 你好小李！很高兴认识你！作为产品经理，你的工作一定充满挑战和创意吧？如果需要讨论产品设计、用户需求或任何相关话题，我随时可以帮忙哦！ 😊
human: 我负责一款电商APP的迭代
ai: 很高兴能为您提供帮助！作为产品经理，您对电商APP的迭代有什么具体方向或问题需要探讨吗？比如用户增长、功能优化、体验提升，或是数据驱动决策等方面？
human: 最近在优化用户下单流程
ai: 好的，小李。优化下单流程是提升转化率和用户体验的关键。基于我们之前的讨论，这里有几个核心方向和具体建议供你参考：
...省略

需要进一步讨论具体功能或数据指标吗？

最终对话摘要：小李是电商APP产品经理，正优化下单流程以解决用户流失率高的问题，需要具体优化建议。
```

仔细观察发现，记忆里不是逐字逐句的对话原文，而是一段总结。这样即使对话很多，摘要也不会太长，非常适合超长对话场景。

#### 3.1.2.4 三种Memory怎么选？

学完三种模式后，大家可能会纠结“该用哪个？”，这里整理了一张对比表，一看就懂：

| 记忆模式 | 核心优势                                | 局限性                                    | 适用场景                                                 |
| :------- | :-------------------------------------- | :---------------------------------------- | :------------------------------------------------------- |
| 全量记忆 | 上下文完整，无信息丢失，实现简单        | Token消耗随轮数线性增长，不适用于长对话   | 短对话、需要完整上下文的场景（如一对一咨询）             |
| 窗口记忆 | Token消耗可控，性能稳定，实现难度低     | 可能丢失早期关键信息，上下文连贯性有限    | 中长对话、对早期信息要求不高的场景（如闲聊）             |
| 摘要记忆 | Token消耗低，支持超长对话，保留核心信息 | 额外消耗LLM算力生成摘要，可能丢失细节信息 | 超长对话、需要平衡上下文与效率的场景（如客服、长期协作） |

#### 3.1.2.5 工程建议

- **存储优化**：示例中使用内存存储（InMemoryChatMessageHistory），生产环境需替换为持久化存储（如Redis、PostgreSQL、MongoDB），基于 `BaseChatMessageHistory` 实现自定义存储类。
- **性能优化**：摘要记忆可缓存摘要结果，避免每次调用都重新生成；窗口记忆可预计算历史消息长度，精准控制Token上限。
- **多模型适配**：可替换LLM为开源模型（如Llama 3、Mistral），降低API调用成本。
- **错误处理**：添加会话历史清理、异常重试、Token溢出检测等逻辑，提升系统稳定性。

#### 3.1.2.6 深入理解：记忆是怎么注入到对话里的？

可能有同学会好奇：“记忆到底是怎么被加到对话里的？” 

核心链路其实很简单：`用户新问题 → 记忆组件提取历史对话 → 把“历史+新问题”拼起来 → 发给LLM → LLM生成回复 → 把“新问题+回复”存到记忆里 → 输出结果`

要实现这个链路，需要两个核心组件配合：

1. ChatMessageHistory：相当于“记忆笔记本”，负责具体的存和取操作；
2. RunnableWithMessageHistory：相当于“记忆调度员”，负责协调整个流程——在调用LLM前自动取历史，调用后自动存新对话。

知道了原理，可以自己动手实践，不借助langchain的框架，自己实现全量记忆

## 3.2外部行动层（Tool）：让AI能“动手”解决问题

学完记忆组件，AI已经能记住我们说的话了，但还有一个局限：它的知识只停留在训练数据里，没法获取实时信息（比如今天的天气）、操作电脑文件，也没法调用其他API。而Tool组件，就是给AI装上“手和脚”，让它能调用外部工具，解决这些原生能力解决不了的问题。