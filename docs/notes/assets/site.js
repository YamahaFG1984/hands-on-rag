/* ================================================================
   Hands-On RAG 学习笔记 · 共享脚本：侧边栏、目录、代码高亮、主题
   ================================================================ */
// 保持经典 <script> 而不是 type="module"：模块脚本在 file:// 下会被浏览器拦截，
// 而这些页面经常是直接双击打开的。
(() => {
  'use strict';

  const CHAPTERS = [
    { n: 1,  t: 'RAG 到底是什么？',        d: '开卷考试的比喻、R 与 G，以及第一条最小 RAG 流水线', part: '第一部分 · 打地基' },
    { n: 2,  t: 'RAG 的基础技术栈',        d: '文档解析、切分、嵌入模型、向量数据库与生成' },
    { n: 3,  t: '扩展到生产规模',          d: '高级摄取、混合检索与重排序、护栏与幻觉控制', part: '第二部分 · 走向生产' },
    { n: 4,  t: '部署到生产环境',          d: '生产环境的挑战、参考架构与从 POC 到上线' },
    { n: 5,  t: 'RAG 平台',               d: 'DIY 还是平台、RAG 蔓延、部署选项与 Vectara 实例' },
    { n: 6,  t: '评估你的 RAG 应用',       d: 'LLM-as-a-Judge、检索与生成指标、在线评估' },
    { n: 7,  t: '从 RAG 到 AI 智能体',     d: '智能体循环、工具调用、MCP、A2A 与智能体记忆', part: '第三部分 · 超越文本检索' },
    { n: 8,  t: '多模态 RAG',             d: '文档里的表格与图像、音频与视频' },
    { n: 9,  t: '知识增强 RAG',           d: '知识图谱、GraphRAG、图数据库与准确率/成本权衡' },
    { n: 10, t: 'RAG 的未来',             d: 'Agentic RAG、Context 工程、边缘小模型与治理' }
  ];

  const pad = (n) => String(n).padStart(2, '0');
  const href = (n) => `ch${pad(n)}.html`;

  const cur = Number.parseInt(document.body.dataset.chapter || '0', 10);

  /* ---------- 侧边栏 ---------- */
  const side = document.getElementById('sidebar');
  if (side) {
    const brand = '<a class="brand" href="index.html"><span class="mark">R</span>' +
      '<span>Hands-On RAG 笔记<small>从原型到生产的检索增强生成</small></span></a>';
    const links = CHAPTERS.map(({ n, t, part }) =>
      (part ? `<div class="part">${part}</div>` : '') +
      `<a class="ch${n === cur ? ' active' : ''}" href="${href(n)}"><span class="n">${n}</span><span>${t}</span></a>`
    );
    const ext = '<div class="ext">' +
      '<a href="../">原书介绍页 &rarr;</a>' +
      '<a href="https://github.com/ofermend/hands-on-rag">配套代码仓库 &rarr;</a></div>';
    side.innerHTML = brand + links.join('') + ext;
    const active = side.querySelector('a.ch.active');
    if (active) setTimeout(() => active.scrollIntoView({ block: 'center' }), 0);
  }

  /* ---------- 移动端菜单 ---------- */
  const btn = Object.assign(document.createElement('button'), {
    id: 'menu-btn',
    type: 'button',
    innerHTML: '&#9776;',
    onclick: () => document.body.classList.toggle('nav-open')
  });
  btn.setAttribute('aria-label', '目录');
  document.body.appendChild(btn);
  document.addEventListener('click', (e) => {
    if (document.body.classList.contains('nav-open') &&
        side && !side.contains(e.target) && e.target !== btn) {
      document.body.classList.remove('nav-open');
    }
  });

  /* ---------- 主题切换 ---------- */
  // 页面 <head> 里的内联脚本已经提前恢复了保存的主题，这里只负责切换
  const THEME_KEY = 'rag-notes-theme';
  const tbtn = Object.assign(document.createElement('button'), {
    id: 'theme-btn',
    type: 'button',
    innerHTML: '&#9789;',
    onclick: () => {
      const root = document.documentElement;
      const now = root.getAttribute('data-theme');
      const dark = now ? now === 'dark'
        : window.matchMedia('(prefers-color-scheme: dark)').matches;
      const next = dark ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem(THEME_KEY, next); } catch {}
    }
  });
  tbtn.setAttribute('aria-label', '切换深浅色');
  document.body.appendChild(tbtn);

  /* ---------- 章节内目录 ---------- */
  const main = document.querySelector('main');
  const slot = document.getElementById('chapter-toc');
  if (slot && main) {
    const hs = [...main.querySelectorAll('h2')];
    if (hs.length > 2) {
      const items = hs.map((h, i) => {
        if (!h.id) h.id = `sec-${i + 1}`;
        return `<li><a href="#${h.id}">${h.textContent}</a></li>`;
      });
      slot.className = 'toc';
      slot.innerHTML = `<div class="h">本章目录</div><ul>${items.join('')}</ul>`;
    }
  }

  /* ---------- 代码块：语言标签 + 复制 + 高亮 ---------- */
  const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  const PY_KW = ('def|class|return|if|elif|else|for|while|in|not|and|or|is|import|from|as|with|' +
    'try|except|finally|raise|lambda|yield|async|await|pass|break|continue|global|nonlocal|' +
    'assert|del|None|True|False|self').split('|');

  const CY_KW = ('MATCH|OPTIONAL|WHERE|RETURN|CREATE|MERGE|SET|DELETE|DETACH|WITH|UNWIND|ORDER|BY|' +
    'LIMIT|SKIP|AS|AND|OR|NOT|IN|IS|NULL|DISTINCT|CALL|YIELD|ON|DESC|ASC|COUNT|' +
    'SELECT|PREFIX|FILTER|GROUP|UNION|ASK|CONSTRUCT|DESCRIBE').split('|');

  // 每一段用正则字面量写，取 .source 拼起来，省掉字符串里的双重转义
  const RE_PY = new RegExp([
    /(#[^\n]*)/.source,                                                                         // 1 注释
    /([rfbu]{0,2}(?:"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^\\"\n])*"|'(?:\\.|[^\\'\n])*'))/.source, // 2 字符串
    String.raw`\b(${PY_KW.join('|')})\b`,                                                       // 3 关键字
    /\b([A-Z][A-Za-z0-9_]*)\b/.source,                                                          // 4 类/常量
    /\b(\d+(?:\.\d+)?)\b/.source,                                                               // 5 数字
    /\b([a-zA-Z_][\w]*)(?=\()/.source                                                           // 6 函数调用
  ].join('|'), 'g');

  // Cypher 用 // 注释，SPARQL 用 # 注释，共用一套规则
  const RE_QL = new RegExp([
    /(\/\/[^\n]*|#[^\n]*)/.source,                                  // 1 注释
    /("(?:\\.|[^\\"\n])*"|'(?:\\.|[^\\'\n])*')/.source,             // 2 字符串
    String.raw`\b(${CY_KW.join('|')})\b`,                           // 3 关键字
    /(:[A-Z][A-Za-z0-9_]*)/.source,                                 // 4 标签 / 关系类型
    /\b(\d+(?:\.\d+)?)\b/.source,                                   // 5 数字
    /(\?[A-Za-z_]\w*)/.source                                       // 6 SPARQL 变量
  ].join('|'), 'g');

  const RE_SH = /(#[^\n]*)|('(?:\\[\s\S]|[^\\'])*'|"(?:\\[\s\S]|[^\\"])*")|\b(pip|python|git|cd|docker|export|curl)\b/g;

  const CLASSES = ['tk-cm', 'tk-st', 'tk-kw', 'tk-tp', 'tk-nm', 'tk-fn'];
  const RULES = { python: RE_PY, cypher: RE_QL, sparql: RE_QL, bash: RE_SH, sh: RE_SH };
  const LABELS = { text: '示意', python: 'Python', cypher: 'Cypher', sparql: 'SPARQL', bash: 'Shell' };

  function paint(src, re) {
    let out = '';
    let last = 0;
    // 所有分支都至少匹配一个字符，matchAll 不会卡在零宽匹配上
    for (const m of src.matchAll(re)) {
      out += esc(src.slice(last, m.index));
      const g = m.findIndex((v, i) => i > 0 && v !== undefined);
      out += `<span class="${CLASSES[g - 1]}">${esc(m[g])}</span>`;
      last = m.index + m[0].length;
    }
    return out + esc(src.slice(last));
  }

  document.querySelectorAll('.code').forEach((box) => {
    const pre = box.querySelector('pre');
    if (!pre) return;
    const lang = box.dataset.lang || 'text';

    const bar = document.createElement('div');
    bar.className = 'bar';
    bar.innerHTML = `<span class="tag">${esc(LABELS[lang] || lang)}</span>`;
    const cp = Object.assign(document.createElement('button'), {
      className: 'copy',
      type: 'button',
      textContent: '复制'
    });
    cp.onclick = () => {
      navigator.clipboard?.writeText(pre.textContent);
      cp.textContent = '已复制';
      setTimeout(() => { cp.textContent = '复制'; }, 1400);
    };
    bar.appendChild(cp);
    box.insertBefore(bar, pre);

    const code = pre.textContent.replace(/^\n/, '').replace(/\s+$/, '');
    const re = RULES[lang];
    pre.innerHTML = re ? paint(code, re) : esc(code);
  });

  /* ---------- 上一章 / 下一章 ---------- */
  const pager = document.getElementById('pager');
  if (pager && cur) {
    const prev = CHAPTERS.find((c) => c.n === cur - 1);
    const next = CHAPTERS.find((c) => c.n === cur + 1);
    const prevLink = prev
      ? `<a class="prev" href="${href(prev.n)}"><span>&larr; 上一章</span>第 ${prev.n} 章 · ${prev.t}</a>`
      : '<a class="prev" href="index.html"><span>&larr; 返回</span>笔记首页</a>';
    const nextLink = next
      ? `<a class="next" href="${href(next.n)}"><span>下一章 &rarr;</span>第 ${next.n} 章 · ${next.t}</a>`
      : '<a class="next" href="index.html"><span>读完了 &rarr;</span>回到笔记首页</a>';
    pager.className = 'pager';
    pager.innerHTML = prevLink + nextLink;
  }

  /* ---------- 首页目录 ---------- */
  const grid = document.getElementById('toc-grid');
  if (grid) {
    const groups = [];
    CHAPTERS.forEach((c) => {
      if (c.part || !groups.length) groups.push({ part: c.part, items: [] });
      groups[groups.length - 1].items.push(c);
    });
    grid.innerHTML = groups.map(({ part, items }) =>
      `<div class="part-h">${part}</div><div class="toc-grid">` +
      items.map(({ n, t, d }) =>
        `<a class="toc-card" href="${href(n)}">` +
        `<div class="n">第 ${n} 章</div>` +
        `<div class="t">${t}</div>` +
        `<div class="d">${d}</div></a>`
      ).join('') + '</div>'
    ).join('');
  }
})();
