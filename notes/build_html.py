"""把 notes/chapterN.md 渲染成 docs/notes/chNN.html，供 GitHub Pages 展示。

用法（在仓库根目录）：
    pip install markdown-it-py
    python notes/build_html.py

样式和交互（侧边栏、目录、高亮、深浅色）都在 docs/notes/assets/ 里，
这里只负责 Markdown → HTML 和少量结构调整。
"""

import html
import re
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "notes"
OUT = ROOT / "docs" / "notes"

# 在样式表加载前恢复保存的主题，避免深色模式下先闪一下白底
THEME_BOOT = (
    "<script>try{const t=localStorage.getItem('rag-notes-theme');"
    "if(t)document.documentElement.setAttribute('data-theme',t)}catch{}</script>"
)


def page(title, chapter, body):
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
{THEME_BOOT}
<link rel="stylesheet" href="assets/style.css">
</head>
<body data-chapter="{chapter}">
<div id="app">
<aside id="sidebar"></aside>
<main>
{body}
</main>
</div>
<script src="assets/site.js"></script>
</body>
</html>
"""


def render_fence(self, tokens, idx, options, env):
    tok = tokens[idx]
    lang = tok.info.strip().split()[0] if tok.info.strip() else "text"
    return (f'<div class="code" data-lang="{html.escape(lang)}">'
            f"<pre>{html.escape(tok.content)}</pre></div>\n")


md = MarkdownIt("commonmark", {"html": False}).enable("table").enable("strikethrough")
md.add_render_rule("fence", render_fence)

# CommonMark 不认紧贴中文标点的 **，比如 "**宾语（object）**三部分"，还会把后面的
# 粗体配错对。所以先在源文本里逐行配对，换成私用区占位符，渲染后再换回 <strong>。
# 代码块和行内代码里的 ** 原样保留。
B_OPEN, B_CLOSE = "\ue000", "\ue001"


def mark_bold(text):
    out, fence = [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            fence = not fence
        elif not fence:
            parts = re.split(r"(`[^`]*`)", line)
            parts[::2] = [re.sub(r"\*\*(.+?)\*\*", B_OPEN + r"\1" + B_CLOSE, x) for x in parts[::2]]
            line = "".join(parts)
        out.append(line)
    return "\n".join(out)


def render(text, inline=False):
    out = md.renderInline(mark_bold(text)) if inline else md.render(mark_bold(text))
    return out.replace(B_OPEN, "<strong>").replace(B_CLOSE, "</strong>")


def build_chapter(n):
    text = (SRC / f"chapter{n}.md").read_text(encoding="utf-8")

    # 第一行是 "# 第X章教学笔记：标题"，第一段（全角括号）是出处说明
    lines = text.split("\n")
    h1 = lines[0].lstrip("# ").strip()
    title = h1.split("：", 1)[1] if "：" in h1 else h1
    rest = "\n".join(lines[1:]).lstrip()
    lede = ""
    if rest.startswith("（"):
        lede, rest = rest.split("\n", 1)
        lede = lede.strip().strip("（）")

    # 各节之间的 --- 分隔线由 h2 的上边框代替
    rest = re.sub(r"^---\s*$", "", rest, flags=re.M)

    body = render(rest)
    body = re.sub(r"<table>", '<div class="tw"><table>', body)
    body = re.sub(r"</table>", "</table></div>", body)
    body = body.replace("<blockquote>", '<div class="box">').replace("</blockquote>", "</div>")
    sec = iter(range(1, 100))
    body = re.sub(r"<h2>", lambda _: f'<h2 id="sec-{next(sec)}">', body)

    head = (f'<span class="chapter-eyebrow">第 {n} 章</span>\n'
            f"<h1>{render(title, inline=True)}</h1>\n")
    if lede:
        head += f'<p class="lede">{render(lede, inline=True)}</p>\n'
    head += '<div id="chapter-toc"></div>\n'

    doc = page(f"第 {n} 章 · {title} · Hands-On RAG 学习笔记", n,
               head + body + '<nav id="pager"></nav>')
    (OUT / f"ch{n:02d}.html").write_text(doc, encoding="utf-8")


INDEX_BODY = """
<div class="hero">
  <span class="chapter-eyebrow">10 章中文学习笔记</span>
  <h1>Hands-On RAG for Production 学习笔记</h1>
  <p class="sub">
    这是 O'Reilly《Hands-On RAG for Production》（Ofer Mendelevitch、Forrest Sheng Bao 著）的逐章中文笔记。
    每一章都对照原书正文和仓库里的 notebook 整理：先用大白话讲清楚概念为什么存在，再拆解代码，
    最后用术语表收尾 —— 从一条最小的 RAG 流水线，一路走到评估、智能体、多模态与知识图谱。
  </p>
  <p style="margin-top:14px">
    <span class="pill">LangChain</span><span class="pill">LanceDB</span><span class="pill">混合检索</span>
    <span class="pill">重排序</span><span class="pill">护栏</span><span class="pill">LLM-as-a-Judge</span>
    <span class="pill">Vectara</span><span class="pill">MCP</span><span class="pill">多模态</span>
    <span class="pill">Neo4j</span><span class="pill">GraphRAG</span>
  </p>
</div>

<h2 id="how">怎么读这套笔记</h2>

<div class="cards">
  <div class="card"><span class="ic">&#128218;</span><h4>先讲为什么</h4>
    <p>每个技术点先说明它要解决的问题，再介绍做法，避免只记住名词。</p></div>
  <div class="card"><span class="ic">&#128187;</span><h4>对照代码</h4>
    <p>笔记里的代码片段来自各章 notebook，可以在仓库对应的 <code>chapterN/</code> 目录里找到并运行。</p></div>
  <div class="card"><span class="ic">&#9888;&#65039;</span><h4>生产视角</h4>
    <p>重点标注书里提到的「生产环境会踩的坑」：成本、延迟、权限、评估和幻觉。</p></div>
  <div class="card"><span class="ic">&#128278;</span><h4>术语表</h4>
    <p>每章末尾都有新增术语表，用一句话解释，方便回头查。</p></div>
</div>

<h2 id="contents">目录</h2>

<div id="toc-grid"></div>
"""


def build_index():
    doc = page("Hands-On RAG for Production 学习笔记", 0, INDEX_BODY.strip())
    (OUT / "index.html").write_text(doc, encoding="utf-8")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for n in range(1, 11):
        build_chapter(n)
    build_index()
    print(f"wrote 11 pages to {OUT.relative_to(ROOT)}/")
