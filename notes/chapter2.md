# 第二章教学笔记：RAG 的基础技术栈（The Base RAG Stack）

（依据《Hands-On RAG for Production》第二章原文 + `chapter2/` 下的 notebooks 整理）

第一章我们看到了一个"能跑"的 RAG 例子，但没细讲每一步内部是怎么工作的。第二章就是把 RAG 流水线拆开，一个零件一个零件地讲清楚：**解析（Parsing）→ 切分（Chunking）→ 嵌入（Embedding）→ 存储与检索（Vector DB）→ 生成（LLM）**。

---

## 一、两条流水线：Ingestion Flow 和 Query Flow

RAG 系统本质上只有两条流程：

**摄取流程（Ingestion Flow）**——数据准备阶段，通常在有新数据、或旧数据要更新时才跑一次：

```
数据源(数据库/API/文件/网页) → 解析(Parsing) → 切分(Chunking) → 嵌入(Embedding) → 存入向量库(Vector DB)
```

**查询流程（Query Flow）**——每次用户提问都会触发：

```
用户提问 → 查询改写 → 向量化 → 在向量库里检索相关内容 → (可选)重排序 → 拼装Prompt → LLM生成回答
```

书里特别提到一个细节：**查询改写（query rewriting）**。比如用户说"根据我2025年的旅行经历写一首诗"，这句话其实包含两部分："写一首诗"（是个指令）和"我2025年的旅行经历"（这才是需要去检索的内容）。如果把整句话都拿去检索，可能会搜到一堆"诗歌"相关但跟"旅行"无关的内容。所以要把问题拆开，只用真正需要查资料的那部分去做检索——这部分被称为 **retrieval query（检索查询）**，区别于用户的原始提问。

---

## 二、文档解析（Document Parsing）

### 为什么解析很难？

PDF 骨子里是为了"排版好看"设计的，不是为了"内容结构清晰"设计的。一页 PDF 内部存的其实是一堆"字符 + 它在页面上的二维坐标"，根本没有"这是一个段落""这是一个句子"这种结构信息。所以从 PDF 里把文字、表格、图片分别提取出来，是个不轻松的工程问题。

DOCX、HTML 反而好处理很多，因为它们是 XML 家族的格式，文字和它的属性（字体、样式）是结构化存储在一起的。

### 用 PyMuPDF 解析 PDF（代码示例，对应 `chapter2/parse-pdf.ipynb`）

最简单的方式——直接按页取纯文本：

```python
with pymupdf.open("sample_data/sample_data.pdf") as doc:
    for page in doc:
        text: str = page.get_text()
        print(text, end="---")
```

问题：表格里的文字会和正文文字混在一起，输出很乱。

改进一点：按"块"（block，PyMuPDF自动聚类出来的文本块）取文本：

```python
with pymupdf.open("sample_data/sample_data.pdf") as doc:
    for page in doc:
        text: str = page.get_text("blocks")
        for block in text:
            print(block[4], end="---\n")
```

但还是分不清哪块是表格哪块是正文。真正干净的做法是：先用 `page.find_tables()` 把表格的位置（bounding box）找出来，再用 `page.get_text(clip=bbox)` 单独提取表格内容，最后从全文里把这些表格对应的行过滤掉，剩下的才是"干净正文"：

```python
def extract_unstructured_text(pdf_path):
    """提取PDF中的非表格正文文字"""
    unstructured_text = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            page_text = page.get_text()
            tables = page.find_tables()
            table_text_blocks = []
            if tables.tables:
                for table in tables:
                    bbox = table.bbox
                    table_text = page.get_text(clip=bbox)
                    table_text_blocks.append(table_text.strip())
            lines = page_text.split('\n')
            filtered_lines = []
            for line in lines:
                line = line.strip()
                if line:
                    is_table_content = False
                    for table_text in table_text_blocks:
                        if line in table_text:
                            is_table_content = True
                            break
                    if not is_table_content:
                        filtered_lines.append(line)
            page_unstructured = '\n'.join(filtered_lines)
            if page_unstructured.strip():
                unstructured_text.append(page_unstructured)
    return '\n\n'.join(unstructured_text)
```

图片提取则要用 `page.get_images()`（拿到图片的"引用编号" xref）+ `doc.extract_image(xref)`（拿到真正的二进制图片数据）。

### 用 python-docx 解析 DOCX（对应 `chapter2/parse-docx.ipynb`）

DOCX 本质是个 ZIP 压缩包，正文用 `python-docx` 很容易拿：

```python
from docx import Document
document = Document('sample_data/sample_data.docx')

for para in document.paragraphs:
    print(para.text)

for table in document.tables:
    for row in table.rows:
        row_text = [cell.text for cell in row.cells]
        print(row_text)
```

图片则要靠 `zipfile` 模块把 DOCX 当压缩包打开，找里面的 `.jpg/.png/.gif` 文件读出来。

### 用 LLM 直接解析 PDF（对应 `chapter2/parse-in-llm.ipynb`）

上面手写代码略显繁琐，那能不能让 LLM 直接帮你解析？可以，把文件传给 GPT-5.1，用自然语言下指令就行：

```python
from openai import OpenAI
client = OpenAI()

file = client.files.create(
    file=open("sample_data/sample_data.pdf", "rb"),
    purpose="user_data"
)

completion = client.chat.completions.create(
    model="gpt-5.1",
    messages=[{
        "role": "user",
        "content": [
            {"type": "file", "file": {"file_id": file.id}},
            {"type": "text", "text": "Extract the text content from the file. Exclude texts from tables or images."}
        ]
    }]
)
print(completion.choices[0].message.content)
```

**但书里明确提醒：这个方法要谨慎用。** LLM 生成式的本质决定了它可能"幻觉"，而且现在的 LLM 往往没法真正把文档里嵌入的图片"原样复刻"出来（我们在实际 notebook 里也验证过：让 LLM 尝试提取一张图片转成 base64，结果解码出来根本不是合法图片）。而且对于海量文档，调 LLM 解析又贵又慢。**推荐策略是"先用便宜的传统工具（PDF库/HTML解析器/OCR），实在不行再上 LLM 兜底"。**

---

## 三、文本切分（Chunking）

### 为什么必须切分？三个理由

1. **上下文窗口有限**：虽然现在 LLM 的上下文窗口很大（GPT-5.1: 400k，Claude Sonnet 4.5: 1M，Gemini 3: 1M），听起来很唬人，但换算一下：一个英文单词约等于1.3个token，一个人正常语速是每分钟120词，400k窗口撑死也就装下约42小时的语音转录——面对企业级任务（比如汇总几十份财报电话会议记录）根本不够用。而且**你不需要把整本书都塞给LLM，只需要跟问题相关的那一小部分**，这正是切分+检索的意义。

2. **性能与成本**：书里给了一组真实的延迟数据（Llama 3.3 70B, 2×H100）：

   | Token数 | 延迟 |
   |---|---|
   | 200 | 31ms |
   | 500 | 47ms |
   | 1000 | 82ms |
   | 5000 | 406ms |
   | 10000 | 1833ms |

   Transformer 的推理耗时大致随上下文长度**平方级增长**，塞进去的内容越长，越慢越贵。切分能把喂给 LLM 的内容压缩到刚好够用的量。

3. **提升检索和生成质量**：如果一个大块内容涵盖了好几个话题，那这个块的 embedding 向量就会是这几个话题的"平均值"，对每个话题的代表性都会被稀释。切成小块后，每个块只聚焦一个话题，向量更精准，检索出来的结果相关性更高。同时，LLM 处理长上下文时"推理能力会随上下文变长而下降"（有实验证明），块越小、越精准，LLM 越不容易被无关信息干扰。

### 几种切分策略对比

| 方法 | 优点 | 缺点 |
|---|---|---|
| **固定长度切分**（按字符数硬切） | 实现简单、块大小可预测、便于批处理 | 破坏语义/句子结构，需要靠重叠来补救 |
| **句子/段落切分**（内容感知） | 保留语言结构，块更连贯 | 句子边界检测可能出错（缩写、标点陷阱），块大小不均匀 |
| **递归切分**（如 LangChain 的 `RecursiveCharacterTextSplitter`） | 灵活，能适应多层级结构（段落→句子→子句） | 实现较复杂，仍可能忽略真正的语义边界 |
| **文档结构切分**（按 Markdown 标题层级等） | 与文档逻辑结构对齐 | 依赖良好的文档格式，非结构化文本不适用 |
| **语义切分**（按句子间语义相似度聚类） | 块在语义上更连贯，检索质量理论上更好 | 计算成本高，需要先跑embedding，块大小不可控 |

有意思的是，书里提到一篇 2024 EMNLP 论文（作者之一正是本书作者 Forrest Bao）发现：在 BEIR、RAGBench 这两个业界常用的检索评测基准上，**固定长度切分和语义切分的效果居然没有显著差异**——但作者也提醒，这些基准的文本普遍较短，在真正的长文档场景下结论可能不同。**结论：没有万能的切分策略，最好的办法是拿自己的数据做评测。**

### 代码演示：用 spaCy 做句子切分（对应 `chapter2/spacy_sentencizer.py`）

```python
import spacy
nlp = spacy.load("en_core_web_sm")

text = """Mr. Wang is a teacher. He teaches A.I. (?). Does he love his work?
Of course!"""
doc = nlp(text)

for sent in doc.sents:
    print(sent.text)
```

输出：
```
Mr. Wang is a teacher.
He teaches A.I. (?).
Does he love his work?
Of course!
```

**这个例子的教学意义**：句子里有"Mr."、"A.I."这种带句号的缩写，还有句子中间嵌了个问号 "(?)"，如果只是简单粗暴地按"句号/问号"切分，很容易把一句话切错。spaCy 的 sentencizer 能正确处理这些边界情况——这就是为什么书里说"简单粗暴的正则切分不可靠，用专门的句子分割工具更好"。

对比一下固定长度切分的"暴力效果"：

```python
chunk_size = 30
chunk_overlap = 8
step_size = chunk_size - chunk_overlap
for i in range(0, len(text), step_size):
    chunk = text[i:i+chunk_size]
    print(chunk)
```

输出会把单词硬生生切断（比如"He teac" / " He teaches A.I. (?). Does he"），这是固定长度切分最大的问题——牺牲了语义完整性换取实现简单和速度。

---

## 四、嵌入模型（Embedding Models）

### 为什么需要 Embedding？

设想你要在文档里找所有跟"United States"相关的内容。如果只靠字符串匹配，你会漏掉写成"the U.S."或"America"的段落——它们说的是同一件事，但字面完全不同。**这就是"字面相似度"和"语义相似度"不对齐的问题**，是自然语言处理里的老大难问题。

Embedding 模型的作用，就是把一段文字转换成一串数字（向量），让**语义相近的文字，转出来的数字串在空间里的位置也相近**。这样"the U.S."和"America"的向量会很接近，即使它们的字符串完全不同。

（这个技术的发展史很有意思：从上世纪90年代 Elman network 的 one-hot 编码，到2013年 Google 的 Word2Vec（"queen"减"woman"约等于"king"减"man"这种向量运算，一度轰动学界），再到2018年 BERT 引入"上下文相关"的动态嵌入——现在的嵌入模型早已能捕捉词语在不同语境下的不同含义。）

### 选型要点

- **模型大小 vs 性能权衡**：可以参考 MTEB（Massive Text Embedding Benchmark）排行榜。
- **嵌入维度**：现在很多模型（如OpenAI、Gemini的嵌入模型）支持用 **MRL（Matryoshka Representation Learning，套娃表征学习）** 技术做维度截断——即可以直接砍掉向量的后半部分维度，同时依然保留大部分语义信息（因为训练时就把最有判别力的信息压到了向量前面）。
- **上下文窗口限制**：嵌入模型也有类似 LLM 的最大输入长度限制（如 OpenAI text-embedding-3系列是8k token），如果你的 chunk 比这个还长，很多框架会**默默截断**，导致信息丢失且不会报错——所以 **切分策略要和嵌入模型的上下文窗口匹配**。

### 代码演示：用 Sentence Transformers 生成嵌入并比较相似度（对应 `chapter2/embedding.ipynb`）

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "I am a happy person.",
    "I am a joyful person.",
    "I am a pessimistic person.",
    "I am not an optimistic person."
]
embeddings = model.encode(sentences)
print(embeddings.shape)  # (4, 384) —— 4句话，每句384维向量

similarity_matrix = model.similarity(embeddings, embeddings)
print(similarity_matrix)
```

结果是个 4×4 相似度矩阵，对角线永远是1（自己和自己100%相似）。有意思的观察：

- 句子1（"happy"）和句子2（"joyful"）相似度 0.8151，很高——语义相近。
- 句子3（"pessimistic"）和句子4（"not optimistic"）相似度 0.7047，也不低——虽然字面完全不同（一个用了否定句），但语义接近。

**一个更有启发性的实验**：书里随机截取向量的一小段维度（比如第67到158维，只占384维中的一小部分），重新计算相似度矩阵——发现**相似度排序的模式基本不变**！这说明现代嵌入模型的语义信息是"分布式"存储的，不是集中在某几个特定维度上，随便截取一部分子空间依然能捕捉到有意义的语义关系。

---

## 五、向量数据库与向量检索（Vector Databases and Vector Search）

### 暴力搜索 vs 近似最近邻（ANN）

有了向量，下一步是"给定一个查询向量，如何快速从几万/几百万个向量里找出最相似的几个"？

最朴素的办法是**暴力搜索**：拿查询向量和库里每一个向量都算一次点积（dot product，向量都归一化后点积就等于余弦相似度），谁的分数高就是谁更相似。复杂度是 O(n)——数据量小的时候完全没问题，直接存内存里算就行；但数据上百万甚至上亿条时就跑不动了。

这时就需要 **ANN（Approximate Nearest Neighbor，近似最近邻）算法**——"近似"的意思是不保证100%找到数学上最近的那个，但能找到一个"足够近"的，用极小的代价换取巨大的速度提升。

**最主流的 ANN 算法叫 HNSW（Hierarchical Navigable Small World，分层可导航小世界）**。书里用了一个特别形象的比喻——想象你要找离某个坐标最近的美国城市：

1. **第一层**：只跟纽约、洛杉矶两个"地标城市"比较，判断大致方向；
2. **第二层**：假设更接近洛杉矶，那就在"洛杉矶、旧金山、西雅图、丹佛"这几个大城市里再比一次；
3. **第三层**：假设更接近旧金山，那就细化到"旧金山、圣何塞、奥克兰"这几个湾区城市里比；
4. **第四层**：最后精确到"帕罗奥图、山景城、桑尼维尔"这几个南湾小城里，找出真正最近的。

这种"先粗后细、逐层收窄"的搜索方式，让 HNSW 只需要检查数据集里极小一部分候选，就能拿到接近精确搜索的结果——这就是它高效的原因。（当然万一在两个大区域的"分界线"上，搜索有可能被带偏，找到的不是数学上绝对最近的那个——这就是"近似"的代价。）

### 向量数据库不只是"存向量"

除了向量索引，向量数据库还要处理：持久化、增删改、**元数据过滤（metadata filtering）**、扩展性、和其他系统集成。比如你有几十年所有上市公司的季度财报，如果只想查"某公司在某个时间段"的数据，先用元数据（公司名、日期）把搜索范围缩小，再做向量搜索，会比对全库暴力搜索快得多、也准得多。

### 检索时的关键参数：k

`k` 就是"每次检索返回几个最相关的结果"（回忆第一章代码里的 `search_kwargs={"k": 3}`）。这个值不是越大越好：
- **k 太小**：真正的答案可能排在第k名之后，被漏掉了（这正是第一章"审判结局"那道题翻车的原因）。
- **k 太大**：可能超过 LLM 的上下文窗口；引入不相关的噪音信息，拉低生成质量；增加不必要的成本和延迟。

### 代码演示：用 pgvector 存储和检索向量（对应 `chapter2/pgvector-simple.ipynb`）

pgvector 是给 PostgreSQL 加装的向量搜索扩展。核心步骤：

```python
import psycopg2

conn = psycopg2.connect(host="YOUR_HOST", port="YOUR_PORT", database="YOUR_DB", user="YOUR_USER")
conn.autocommit = True
cursor = conn.cursor()

cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS sentence_embeddings (
        sentence TEXT,
        embedding VECTOR(384)
    )
""")
# 建立 HNSW 索引，加速相似度检索
cursor.execute("""
    CREATE INDEX ON sentence_embeddings
    USING hnsw (embedding vector_l2_ops)
    WITH (m = 16, ef_construction = 64);
""")
```

插入数据（注意 embedding 要先转成 Python list 再序列化成字符串插入）：

```python
for sentence, embedding in zip(sentences, embeddings):
    embedding_as_list = embedding.tolist()
    cursor.execute(
        "INSERT INTO sentence_embeddings (text, embedding) VALUES (%s, %s::vector)",
        (sentence, embedding_as_list)
    )
```

检索函数（`<=>` 是 pgvector 的余弦距离运算符，注意它算的是"距离"不是"相似度"，所以要用 `1 - 距离` 换算成相似度）：

```python
def vector_search(query, model, top_k):
    query_embedding = model.encode([query])[0]
    query_embedding_as_list = query_embedding.tolist()
    cursor.execute("""
        SELECT text, 1 - (embedding <=> %s::vector) as similarity
        FROM sentence_embeddings
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (query_embedding_as_list, query_embedding_as_list, top_k))
    return cursor.fetchall()
```

试一下查询"I am a smiling person."：

```
('I am a happy person.', 0.7639919010393534)
('I am a joyful person.', 0.6934391466808068)
('I am not an optimistic person.', 0.3835604305136333)
```

可以看到"微笑的人"和"快乐/开心的人"相似度明显更高，和"不乐观的人"相似度显著更低——**符合直觉，说明检索是有效的**。

---

## 六、生成式 LLM（Generative LLMs）—— RAG 流程的最后一步

这一步是把检索到的文字块 + 用户原始问题一起交给 LLM，让它生成最终答案。注意：这一步 LLM 拿到的是**原始文本**，不是向量——向量只是用来"找出相关内容"的中间产物，真正喂给 LLM 的还是人类能读的文字。

### Prompt 设计的几个技巧

1. **query 放在 context 前面**：虽然理论上把改写过的"检索查询"放进 prompt 更贴近用户意图，但书里指出，由于训练数据的习惯，**多数 LLM 在"问题在前、材料在后"的结构下表现更好**：
   ```
   这是搜索问题：{query}
   这是相关材料：{retrieved_results}
   ```
2. **加入思维链（Chain-of-Thought）指令**：比如"如果答案不明显，请一步步推理并给出详细解释"，可以引导模型更好地利用推理能力。
3. **明确要求模型不要编造**：比如"如果无法从材料中合理推断答案，请直接说'我不知道'"——这是控制幻觉最直接的一招。
4. **针对具体任务定制模板**：如果你知道用户主要是问答型需求，就把模板写得更具体（"回答问题{query}，依据材料{retrieved_results}"）；如果主要是总结型需求，就换成"请总结关于{query}的信息"。
5. **用 XML 标签分隔不同部分**：比如 `<query>...</query>` 和 `<context>...</context>`，帮助模型更清楚地区分 prompt 里的不同角色。

### 如何评价 LLM 生成的好坏？—— 先预告一下 LLM-as-a-Judge

书里在这里先简单提了一下（第六章会深入讲）：常见做法叫 **LLM-as-a-Judge**——用另一个 LLM 调用，去给生成的回答打分，主要看两个维度：
- **Relevance（相关性）**：回答是否切题、真的回答了用户的问题？
- **Faithfulness（忠实度）**：回答是否真的是依据检索到的材料生成的，有没有编造材料里没有的内容？

这个方法的缺点也很明显：慢、贵、有时候不准（评判用的 LLM 自己也可能推理出错），而且同一个 LLM 对同一个回答，不同时间打分可能不一致。所以还有专门训练的小模型做评判（比如本书作者参与的 Vectara HHEM 模型），以及"人工评估"作为最终、最可信但也最耗时的手段。

### 代码演示：用 Anthropic Claude 生成回答（对应 `chapter2/generative-llms.ipynb`）

这个例子直接拿本章"如何评估LLM"这一节的文字本身作为 context，问 Claude 3 个问题：

```python
import anthropic

prompt_template = """You are a good reader. Answer the query based on the context provided. Give me a short answer.

Query: {query}
Context: {context}"""

response = anthropic.Anthropic().messages.create(
    model="claude-sonnet-4-5",
    messages=[
        {"role": "user", "content": prompt_template.format(query=query, context=context)}
    ]
)
print(response.content[0].text)
```

比如问"如果不考虑预算和时间，评估RAG系统最准的方法是什么？"，Claude 准确从材料里提炼出"人工评估（Human evaluation）是预算和时间不受限时最准确的方式"，并且引用了材料里的原话作为依据——**说明这个简单的 prompt 模板已经能让模型忠实、准确地依据材料回答**。

---

## 七、本章小结：几个"生产环境会踩的坑"

1. **摄取瓶颈（Ingestion bottleneck）**：解析阶段引入的错误（比如OCR识别错、排版信息丢失）会一路传导到后面所有环节，且检索和生成阶段没法把这些错误"修正回来"。
2. **切分策略没有想象中简单**：默认或粗糙的切分方案在生产环境往往不够用，必须靠实际检索效果的评测来验证。
3. **检索本身也有局限**：即使嵌入质量很好，检索依然可能因为"结果冗余""噪音""范围不匹配"而失败——这正是第三章要讲的"混合检索"和"过滤检索"要解决的问题。

**一句话总结第二章**：每一层（解析、切分、嵌入、存储、生成）看起来是独立模块，但实际上环环相扣——检索效果依赖切分质量，生成质量依赖检索准确度，索引设计影响延迟和成本，嵌入模型选择影响相关性。**改进 RAG 系统往往需要端到端地审视这些环节之间的联系，而不是孤立地调优某一环。**

---

## 八、关键术语表（新增）

| 术语 | 大白话解释 |
|---|---|
| Ingestion Flow / Query Flow | 摄取流程（准备数据）和查询流程（响应提问），RAG的两条主线 |
| Query rewriting（查询改写） | 把用户原话改写成更适合去检索的"检索查询" |
| Dense embedding / Sparse vector | 稠密向量（语义嵌入）/ 稀疏向量（关键词式表示，如TF-IDF/BM25） |
| MRL（套娃表征学习） | 一种训练技巧，让嵌入向量可以直接截断到更低维度而不丢失太多语义信息 |
| ANN（近似最近邻） | 用近似算法代替暴力搜索，用极小误差换取巨大速度提升 |
| HNSW | 最主流的ANN算法，分层、由粗到细地逐步逼近最近邻 |
| Metadata filtering（元数据过滤） | 检索前/中/后，用向量之外的字段（如日期、来源）缩小搜索范围 |
| k | 每次检索返回的候选结果数量 |
| LLM-as-a-Judge | 用一个LLM去评判另一个LLM生成结果的相关性和忠实度 |
| Quantization / FlashAttention / vLLM / Ollama | 加速LLM推理速度、降低资源占用的几种常见技术/框架 |

---

## 下一章预告

第三章会讲"如何把这个基础版 RAG 扩展到生产规模"，重点是：处理海量文档、混合检索（语义+关键词）、重排序、护栏（guardrails）、以及幻觉的检测与纠正——正是本章结尾提到的那几个"局限性"的解决方案。
