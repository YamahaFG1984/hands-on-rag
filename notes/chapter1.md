# 第一章教学笔记：RAG 到底是什么？

（依据《Hands-On RAG for Production》第一章原文 + `chapter1/sample-rag.ipynb` 代码整理）

---

## 一、先搞懂问题：LLM 为什么会"睁眼说瞎话"

书里开篇举了个例子：一个工程师用 GPT-5.1 做了个公司内部客服机器人，聊闲天很流畅，但一问"下个月的退款政策是什么"，模型就开始编——因为退款政策这种私有数据，**根本没在它训练时见过**。

这不是模型笨，而是它"眼睛看不见"你的私有数据。LLM 的知识全部来自训练时喂给它的公开语料（书、网页、代码等），这些知识被压缩进模型的参数里，术语叫 **parametric knowledge（参数化知识）**。一旦训练结束，它就是"死"的——不知道你公司昨天更新的政策，也不知道你数据库里的数据。

那能不能训练时把所有私有数据也塞进去？不现实：数据一直在变，你不可能每天重新训练一次模型。

**RAG（Retrieval-Augmented Generation，检索增强生成）就是为了解决这个问题而生的。**

---

## 二、RAG 的核心比喻：开卷考试 vs 闭卷考试

| | 闭卷考试 | 开卷考试 |
|---|---|---|
| 对应 | 直接问 LLM（无 RAG） | RAG |
| 依赖 | 只能凭脑子里记住的知识答题 | 可以翻书、查资料再答题 |

普通问 LLM，就是"闭卷考试"——它只能凭训练时记住的东西回答，记错了/没学过就瞎编。

RAG 相当于让 LLM"开卷考试"：回答问题前，先让它去翻一下"教材"（你的私有数据），把相关的那几页找出来，再照着这几页内容组织答案。

---

## 三、RAG 的两个字母：R 和 G

```
用户提问
   │
   ▼
【R = Retrieval 检索】── 去你的私有数据里，找出和问题最相关的那几段文字（叫 "facts" 或 "chunks"）
   │
   ▼
【G = Generation 生成】── 把"用户的问题" + "刚检索到的那几段文字" 一起丢给 LLM，让它只根据这些材料来回答
   │
   ▼
最终答案
```

书里给出了一个极简的 prompt 模板：

```
你是一个问答助手。请仅使用下面提供的检索到的上下文来回答问题。
如果你不知道答案，就直说不知道。

<question>
{question}
</question>
<context>
{context}
</context>
回答：
```

`{context}` 是检索出来的那几段"教材内容"，`{question}` 是用户的原始问题。

**为什么这样能减少幻觉？** 因为 LLM 手边已经有真实资料可以照抄/总结，不需要凭空编。检索不到相关资料时，设计良好的 RAG 系统会让模型直接说"不知道"。

---

## 四、为什么不直接把整个文档丢给 LLM？

**方案一："Chat with PDF"**（把全文塞进 prompt）问题：
1. **成本**：内容越多调用越贵，大部分内容跟问题无关，纯浪费。
2. **延迟**：内容越多模型处理越慢。
3. **规模**：企业知识库几十万文件不可能全塞进一个 prompt；"该塞哪些文件"这个问题本身就是"检索"，绕了一圈又回到 RAG。

**方案二：微调（fine-tuning）**（把私有数据继续训练进模型参数）问题：
- 需要深度学习专业知识，容易训练坏（过拟合、引入偏差）；
- 数据一更新就要重新微调，成本高；
- **权限控制做不到**：微调会把不同部门的机密数据全部"揉"进模型权重，没法做到"只有 CEO 能看某些内容"。书里称为**"博格效应"**（《星际迷航》里博格人同化一切，融合了就分不开）。

**结论：RAG 更便宜、更快、更新更及时、权限更可控。**

---

## 五、结合第一章代码，逐步拆解

`chapter1/sample-rag.ipynb` 用《爱丽丝梦游仙境》做演示。

### Step 1：加载 PDF

```python
from langchain_community.document_loaders import PyPDFLoader

pdf_url = "https://www.adobe.com/be_en/active-use/pdf/Alice_in_Wonderland.pdf"
loader = PyPDFLoader(pdf_url)
pages = loader.load()
# Loaded 105 pages from the PDF
```

### Step 2：切分成小块（Chunking）

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # 每块大约1000个字符
    chunk_overlap=200,    # 相邻块之间重叠200字符
    length_function=len,
)
chunks = text_splitter.split_documents(pages)
# Split into 245 chunks
```

为什么切分？语义检索粒度太粗不精准，太大又贵。为什么重叠200字符？避免一句话被切在块边界导致信息腰斩。

### Step 3：生成向量嵌入（Embedding）并存入向量数据库

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import LanceDB

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = LanceDB.from_documents(chunks, embeddings)
# Created vector store with 245 documents
```

**Embedding 大白话**：用模型把一段文字转成一串数字（向量），语义相近的句子转出来的数字串在空间里位置也相近。`LanceDB` 是专门存这些向量、能快速做相似度搜索的数据库。

### Step 4：搭建检索 + 生成的问答链

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}"""
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

- `retriever`：每次提问，从向量库找出语义最相似的 3 个 chunk（R）。
- `format_docs`：把 3 个 chunk 原文拼接成 `{context}`。
- `llm`（`temperature=0`）：G，让模型只根据 context 回答，输出尽量确定不发散。
- `|` 是 LCEL 管道写法：问题 → 组装 context/question → 套 prompt → 调 LLM → 解析成字符串。

### Step 5：提问与翻车案例

Notebook 问了4个问题，前3个（主角/开头、毛毛虫问了什么、疯帽子茶会）都答对了。

**第4个问题"审判结局是什么"答错了**——模型说上下文里没提到结局。原因：检索器 `k=3` 只挑3个最相似的 chunk，审判情节跨越多个 chunk，真正包含"结局"的那段很可能没被选中。**检索没找全，生成端自然给不出正确答案。**

这说明 RAG 的答案质量高度依赖"检索"这一步是否找到了真正相关、完整的内容——这也是后面章节讲"混合检索""重排序"的原因。

---

## 六、关键术语表

| 术语 | 大白话解释 |
|---|---|
| Hallucination（幻觉） | LLM 自信满满地编造一个听起来对、实际上是错的答案 |
| Embedding（向量嵌入） | 把文字转换成一串数字，语义相近的文字数字也相近 |
| Vector DB（向量数据库） | 专门存向量、支持"找出最相似N个"搜索的数据库 |
| Chunk（分块） | 长文档切成的小段文字，是检索和存储的最小单位 |
| Retriever（检索器） | 给一个问题，从向量库找出最相关几段的组件 |
| k | 每次检索返回几个最相关的chunk（本例中是3） |
| Context（上下文） | 检索出来喂给LLM的"证据材料" |
| Parametric knowledge（参数化知识） | LLM训练时记住、封存在模型权重里的知识 |

---

## 七、小结

- RAG = 检索（找资料）+ 生成（照资料回答），目的是让 LLM 能回答训练时没见过的私有/实时数据，同时减少胡编乱造。
- 相比"整份文档塞进prompt"和"微调"，RAG 在成本、延迟、规模、权限控制上都更有优势。
- 第一章的简单实现也暴露了问题（k=3找不全导致答错），后续章节会逐一解决。
