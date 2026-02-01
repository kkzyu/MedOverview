const DATA_BASE = "../data";

const RAG_DEFAULTS = {
  deepseekBaseUrl: "https://api.deepseek.com",
  model: "deepseek-chat",
  topK: 8,
  maxQuestionChars: 2000,
  bm25: { k1: 1.2, b: 0.75 },
};

function byId(id) {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing element #${id}`);
  return el;
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function uniq(arr) {
  return [...new Set(arr)];
}

function tokenize(text) {
  const s = String(text || "").toLowerCase();
  const out = [];
  // 1) ASCII words/numbers
  for (const m of s.matchAll(/[a-z0-9_]+/g)) out.push(m[0]);
  // 2) CJK chars (粗粒度，但足够应付 200~300 篇规模)
  for (const ch of s) {
    if (/\p{Script=Han}/u.test(ch)) out.push(ch);
  }
  return out;
}

function buildPaperDocText(p) {
  const kw = (p.keywords || []).join(" ");
  const tri = p.triple || {};
  return [
    p.title || "",
    kw,
    tri.method || "",
    tri.result || "",
    tri.contribution || "",
    p.abstract || "",
    p.venue || "",
    String(p.year || ""),
  ].join("\n");
}

function buildBm25Index(papers, opts) {
  const { k1, b } = opts;
  const N = papers.length;
  const docIds = papers.map((p) => p.id);
  const docLens = new Array(N);
  const tfs = new Array(N); // Map(term -> tf)

  let totalLen = 0;
  const df = new Map(); // term -> doc freq
  for (let i = 0; i < N; i++) {
    const text = buildPaperDocText(papers[i]);
    const toks = tokenize(text);
    docLens[i] = toks.length;
    totalLen += toks.length;

    const tf = new Map();
    for (const t of toks) tf.set(t, (tf.get(t) || 0) + 1);
    tfs[i] = tf;

    for (const t of tf.keys()) df.set(t, (df.get(t) || 0) + 1);
  }
  const avgdl = totalLen / Math.max(1, N);

  const idf = new Map();
  for (const [t, dfi] of df.entries()) {
    const v = Math.log(1 + (N - dfi + 0.5) / (dfi + 0.5));
    idf.set(t, v);
  }

  function scoreDoc(i, qTerms) {
    const tf = tfs[i];
    const dl = docLens[i] || 0;
    let score = 0;
    for (const t of qTerms) {
      const f = tf.get(t) || 0;
      if (!f) continue;
      const w = idf.get(t) || 0;
      const denom = f + k1 * (1 - b + (b * dl) / Math.max(1e-9, avgdl));
      score += w * ((f * (k1 + 1)) / denom);
    }
    return score;
  }

  function search(query, { allowedIds = null, topK = 8 } = {}) {
    const qTerms = tokenize(query);
    if (!qTerms.length) return [];

    const allowSet = allowedIds ? new Set(allowedIds.map(String)) : null;
    const scored = [];
    for (let i = 0; i < N; i++) {
      const pid = docIds[i];
      if (allowSet && !allowSet.has(String(pid))) continue;
      const s = scoreDoc(i, qTerms);
      if (s > 0) scored.push({ id: pid, score: s });
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK);
  }

  return { search };
}

async function deepseekChatJson({ apiKey, baseUrl, model, system, userObj, maxTokens = 900, temperature = 0.2 }) {
  const url = String(baseUrl || RAG_DEFAULTS.deepseekBaseUrl).replace(/\/+$/, "") + "/v1/chat/completions";
  const resp = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: model || RAG_DEFAULTS.model,
      messages: [
        { role: "system", content: system },
        { role: "user", content: JSON.stringify(userObj) },
      ],
      temperature,
      max_tokens: maxTokens,
      response_format: { type: "json_object" },
    }),
  });
  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(`DeepSeek 请求失败: ${resp.status} ${txt.slice(0, 200)}`);
  }
  const data = await resp.json();
  const content = data?.choices?.[0]?.message?.content || "";
  if (!content) throw new Error("DeepSeek 返回空内容");
  try {
    return JSON.parse(content);
  } catch (e) {
    throw new Error(`DeepSeek 未返回 JSON: ${String(content).slice(0, 200)}`);
  }
}

function buildIndex(papers) {
  const byPid = new Map();
  for (const p of papers) byPid.set(p.id, p);
  return byPid;
}

function getMetaLabel(meta, dim, level, key) {
  // key for l1: l1Id string (e.g. "3" or "-1")
  // key for l2: "l1Id:l2Id" (e.g. "3:1" or "3:-1")
  const d = meta.dimensions?.[dim];
  if (!d) return key;

  if (level === "l1") {
    const x = d.l1?.[String(key)];
    return x?.llm?.name_cn ?? x?.label ?? key;
  }

  // l2
  const [l1, l2] = String(key).split(":");
  const x = d.l2?.[String(l1)]?.[String(l2)];
  return x?.llm?.name_cn ?? x?.label ?? `${l1}:${l2}`;
}

function getDictLabel(dictMap, dim, level, key) {
  if (!dictMap) return null;
  if (level !== "l1") return null;
  const d = dictMap.dimensions?.[dim];
  if (!d) return null;
  const x = d.l1?.[String(key)];
  return x?.type_label_cn ?? null;
}

function getDictDescription(dictMap, dim, level, key) {
  if (!dictMap) return null;
  if (level !== "l1") return null;
  const d = dictMap.dimensions?.[dim];
  if (!d) return null;
  const x = d.l1?.[String(key)];
  return x?.type_description_cn ?? null;
}

function getStageLabel(stageMeta, level, key) {
  if (!stageMeta) return key;
  if (level === "l1") {
    return stageMeta.l1?.[String(key)]?.label ?? key;
  }
  const [l1, l2] = String(key).split(":");
  return stageMeta.l2?.[String(l1)]?.[String(l2)]?.label ?? `${l1}:${l2}`;
}

function paperTextForSearch(p) {
  const kw = (p.keywords || []).join(" ");
  const st = p.stage_text || "";
  return `${p.title} ${kw} ${st} ${p.venue} ${p.year}`.toLowerCase();
}

function makeNodeName(col, dim, kind, key) {
  return `${col}|${dim}|${kind}|${key}`;
}

function parseNodeName(name) {
  const [col, dim, kind, key] = String(name).split("|");
  return { col: Number(col), dim, kind, key };
}

function getPaperDimKeys(labels, pid, dim, kind, expandedL1) {
  // returns list of {kind:'l1'|'l2', key:string}
  if (dim === "stage") {
    const st = labels[pid]?.stage;
    const l1s = Array.isArray(st?.l1) ? st.l1 : [];
    const l2s = Array.isArray(st?.l2) ? st.l2 : [];

    if (kind === "l1") {
      return (l1s.length ? l1s : ["Other"]).map((x) => ({ kind: "l1", key: String(x) }));
    }

    // mixed expansion: if expandedL1 is set and the paper has that parent, show its L2(s)
    const parent = expandedL1 != null ? String(expandedL1) : null;
    if (parent && l1s.map(String).includes(parent)) {
      const children = l2s
        .filter((x) => typeof x === "string" && x.startsWith(parent + "::"))
        .map((x) => x.split("::")[1])
        .filter(Boolean);
      const uniqChildren = children.length ? uniq(children) : ["Other"];
      return uniqChildren.map((c) => ({ kind: "l2", key: `${parent}:${String(c)}` }));
    }
    return (l1s.length ? l1s : ["Other"]).map((x) => ({ kind: "l1", key: String(x) }));
  }

  const l1 = labels[pid]?.[dim]?.l1;
  const l2 = labels[pid]?.[dim]?.l2;

  if (kind === "l1") {
    return [{ kind: "l1", key: String(l1 ?? -1) }];
  }

  const l1s = String(l1 ?? -1);
  if (expandedL1 != null && l1s === String(expandedL1)) {
    return [{ kind: "l2", key: `${l1s}:${String(l2 ?? -1)}` }];
  }
  return [{ kind: "l1", key: l1s }];
}

function venueKey(v) {
  const s = String(v || "").toLowerCase();
  if (s.includes("iclr")) return "ICLR";
  if (s.includes("icml")) return "ICML";
  if (s.includes("neurips") || s.includes("nips")) return "NeurIPS";
  return String(v || "Other");
}

function paperMatchesExpanded(labels, pid, dim, expandedL1) {
  if (expandedL1 == null) return true;
  const parent = String(expandedL1);

  if (dim === "stage") {
    const st = labels?.[pid]?.stage;
    const l1s = Array.isArray(st?.l1) ? st.l1.map(String) : [];
    return l1s.includes(parent);
  }

  const l1 = labels?.[pid]?.[dim]?.l1;
  return String(l1 ?? -1) === parent;
}

function buildSankey({ papers, labels, meta, stageMeta, columns, expanded }) {
  // columns: [dimA, dimB, dimC]
  // expanded: {0: l1Id|null, 1: l1Id|null, 2: l1Id|null}

  // Focus-mode expansion: if a column is expanded to a specific L1, we only keep papers
  // that belong to that parent cluster (so L2 won't be mixed with other L1 nodes).
  const focusPapers = papers.filter((p) => {
    return (
      paperMatchesExpanded(labels, p.id, columns[0], expanded[0]) &&
      paperMatchesExpanded(labels, p.id, columns[1], expanded[1]) &&
      paperMatchesExpanded(labels, p.id, columns[2], expanded[2])
    );
  });

  const nodeCounts = new Map(); // nodeName -> count
  const nodePaperIds = new Map(); // nodeName -> Set(paperId)

  const linkCounts = new Map(); // linkKey -> count
  const linkPaperIds = new Map(); // linkKey -> Set(paperId)

  const nodeLabelMap = new Map(); // nodeName -> label

  function addNode(name, label, pid) {
    nodeCounts.set(name, (nodeCounts.get(name) || 0) + 1);
    if (!nodePaperIds.has(name)) nodePaperIds.set(name, new Set());
    nodePaperIds.get(name).add(pid);
    nodeLabelMap.set(name, label);
  }

  function addLink(source, target, pid) {
    const k = `${source}->${target}`;
    linkCounts.set(k, (linkCounts.get(k) || 0) + 1);
    if (!linkPaperIds.has(k)) linkPaperIds.set(k, new Set());
    linkPaperIds.get(k).add(pid);
  }

  for (const p of focusPapers) {
    const pid = p.id;

    const aList = getPaperDimKeys(labels, pid, columns[0], "mixed", expanded[0]);
    const bList = getPaperDimKeys(labels, pid, columns[1], "mixed", expanded[1]);
    const cList = getPaperDimKeys(labels, pid, columns[2], "mixed", expanded[2]);

    for (const a of aList) {
      for (const b of bList) {
        for (const c of cList) {
          const aName = makeNodeName(0, columns[0], a.kind, a.key);
          const bName = makeNodeName(1, columns[1], b.kind, b.key);
          const cName = makeNodeName(2, columns[2], c.kind, c.key);

          const aLabel = columns[0] === "stage" ? getStageLabel(stageMeta, a.kind, a.key) : getMetaLabel(meta, columns[0], a.kind, a.key);
          const bLabel = columns[1] === "stage" ? getStageLabel(stageMeta, b.kind, b.key) : getMetaLabel(meta, columns[1], b.kind, b.key);
          const cLabel = columns[2] === "stage" ? getStageLabel(stageMeta, c.kind, c.key) : getMetaLabel(meta, columns[2], c.kind, c.key);

          addNode(aName, aLabel, pid);
          addNode(bName, bLabel, pid);
          addNode(cName, cLabel, pid);

          addLink(aName, bName, pid);
          addLink(bName, cName, pid);
        }
      }
    }
  }

  const nodes = [];
  for (const [name, value] of nodeCounts.entries()) {
    nodes.push({ name, value });
  }

  const links = [];
  for (const [k, value] of linkCounts.entries()) {
    const [source, target] = k.split("->");
    links.push({ source, target, value });
  }

  return {
    nodes,
    links,
    nodePaperIds,
    linkPaperIds,
    nodeLabelMap,
  };
}

function renderDetails({ paperIndex, paperIds, searchTerm }) {
  const details = byId("details");
  const detailsHint = document.getElementById("detailsHint");
  const list = (paperIds || []).map((id) => paperIndex.get(id)).filter(Boolean);

  const q = (searchTerm || "").trim().toLowerCase();
  const filtered = q ? list.filter((p) => paperTextForSearch(p).includes(q)) : list;

  if (detailsHint) {
    const shown = Math.min(200, filtered.length);
    detailsHint.textContent = `当前筛选论文：${filtered.length} 篇（展示前 ${shown} 篇）`;
  }

  if (filtered.length === 0) {
    details.innerHTML = `<div class="muted">无匹配论文</div>`;
    return;
  }

  const html = filtered
    .sort((a, b) => (b.year - a.year) || String(a.title).localeCompare(String(b.title)))
    .slice(0, 200)
    .map((p) => {
      const kws = (p.keywords || []).slice(0, 12);
      const triple = p.triple || {};
      const stage = p.stage || null;
      const stageL1 = Array.isArray(stage?.l1) ? stage.l1 : [];
      const stageL2 = Array.isArray(stage?.l2) ? stage.l2 : [];
      const stageTags = [...stageL1, ...stageL2].slice(0, 10);
      return `
        <div class="card">
          <div class="card-title">
            <a href="${escapeHtml(p.url || "#")}" target="_blank" rel="noreferrer">${escapeHtml(p.title)}</a>
          </div>
          <div class="card-meta">${escapeHtml(p.venue || "")} · ${escapeHtml(p.year)} · <span class="muted">${escapeHtml(p.id)}</span></div>
          <div class="kws">${kws.map((k) => `<span class="tag">${escapeHtml(k)}</span>`).join("")}</div>
          ${stageTags.length ? `<div class="kws">${stageTags.map((k) => `<span class="tag">${escapeHtml(k)}</span>`).join("")}</div>` : ""}
          <div class="triple">
            <div><span class="muted">Method:</span> ${escapeHtml(triple.method || "")}</div>
            <div><span class="muted">Result:</span> ${escapeHtml(triple.result || "")}</div>
            <div><span class="muted">Contribution:</span> ${escapeHtml(triple.contribution || "")}</div>
          </div>
          <div class="abstract"><span class="muted">Abstract:</span> ${escapeHtml(p.abstract || "")}</div>
        </div>
      `;
    })
    .join("");

  details.innerHTML = html;
}

async function loadJson(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`fetch failed: ${path}`);
  return res.json();
}

async function main() {
  const footer = document.getElementById("footerCounts");
  if (footer) footer.textContent = "加载数据中…";

  const tryLoadMeta = async () => {
    try {
      return await loadJson(`${DATA_BASE}/cluster_meta_llm.json`);
    } catch {
      return await loadJson(`${DATA_BASE}/cluster_meta.json`);
    }
  };

  const tryLoadStageMeta = async () => {
    try {
      return await loadJson(`${DATA_BASE}/stage_meta.json`);
    } catch {
      return null;
    }
  };

  const tryLoadDictMap = async () => {
    try {
      return await loadJson(`${DATA_BASE}/cluster_label_dict_map.json`);
    } catch {
      return null;
    }
  };

  const [papers, labels, meta, stageMeta, cfg, dictMap, presets] = await Promise.all([
    loadJson(`${DATA_BASE}/papers.json`),
    loadJson(`${DATA_BASE}/labels.json`),
    tryLoadMeta(),
    tryLoadStageMeta(),
    loadJson(`${DATA_BASE}/app_config.json`),
    tryLoadDictMap(),
    (async () => {
      try {
        return await loadJson(`${DATA_BASE}/presets.json`);
      } catch {
        return [];
      }
    })(),
  ]);

  // attach stage labels onto paper objects for rendering/search
  for (const p of papers) {
    const st = labels?.[p.id]?.stage;
    if (st) {
      p.stage = st;
      const l1 = Array.isArray(st.l1) ? st.l1.join(" ") : "";
      const l2 = Array.isArray(st.l2) ? st.l2.join(" ") : "";
      p.stage_text = `${l1} ${l2}`.trim();
    } else {
      p.stage = null;
      p.stage_text = "";
    }
  }

  const paperIndex = buildIndex(papers);

  // RAG index (client-side retrieval)
  const bm25 = buildBm25Index(papers, RAG_DEFAULTS.bm25);

  // Fixed 3 columns as requested.
  const columns = stageMeta ? ["stage", "method", "result"] : ["method", "result", "contribution"];
  const col0label = document.getElementById("col0label");
  const col1label = document.getElementById("col1label");
  const col2label = document.getElementById("col2label");
  if (col0label) col0label.textContent = columns[0];
  if (col1label) col1label.textContent = columns[1];
  if (col2label) col2label.textContent = columns[2];

  const chart = echarts.init(byId("chart"), null, { renderer: "canvas" });

  let expanded = { 0: null, 1: null, 2: null };
  let currentPaperIds = papers.map((p) => p.id);
  let currentContextText = "全量";
  let viewMode = "dict"; // 'original' | 'dict'

  // ===== RAG UI =====
  const ragKey = document.getElementById("ragKey");
  const ragSaveKey = document.getElementById("ragSaveKey");
  const ragClearKey = document.getElementById("ragClearKey");
  const ragQuestion = document.getElementById("ragQuestion");
  const ragAsk = document.getElementById("ragAsk");
  const ragPresets = document.getElementById("ragPresets");
  const ragStatus = document.getElementById("ragStatus");
  const ragAnswer = document.getElementById("ragAnswer");
  const ragCitations = document.getElementById("ragCitations");
  const ragScope = document.getElementById("ragScope");

  function getRagStyle() {
    const el = document.querySelector('input[name="ragStyle"]:checked');
    return el?.value === "cite" ? "cite" : "overview";
  }

  function setRagStatus(s) {
    if (ragStatus) ragStatus.textContent = s || "";
  }

  function setRagAnswerText(t) {
    if (ragAnswer) ragAnswer.textContent = t || "";
  }

  function setRagCitations(citeIds) {
    if (!ragCitations) return;
    ragCitations.innerHTML = "";
    const ids = (citeIds || []).map(String);
    for (const pid of ids) {
      const p = paperIndex.get(pid);
      if (!p) continue;
      const a = document.createElement("a");
      a.className = "qa-cite";
      a.href = p.url || "#";
      a.target = "_blank";
      a.rel = "noreferrer";
      a.textContent = `${p.venue || ""} ${p.year || ""} · ${pid}`.trim();
      a.title = p.title || pid;
      ragCitations.appendChild(a);
    }
  }

  function updateRagScope() {
    if (!ragScope) return;
    ragScope.textContent = `检索范围：${currentPaperIds.length} 篇`;
  }

  function loadSavedKey() {
    try {
      const v = localStorage.getItem("medoverview_deepseek_key");
      if (ragKey && v) ragKey.value = v;
    } catch {
      // ignore
    }
  }

  function saveKey() {
    if (!ragKey) return;
    const v = String(ragKey.value || "").trim();
    if (!v) return;
    localStorage.setItem("medoverview_deepseek_key", v);
  }

  function clearKey() {
    if (ragKey) ragKey.value = "";
    try {
      localStorage.removeItem("medoverview_deepseek_key");
    } catch {
      // ignore
    }
  }

  async function ragRun() {
    if (!ragQuestion) return;
    const question = String(ragQuestion.value || "").trim();
    if (!question) {
      setRagStatus("请输入问题");
      return;
    }
    if (question.length > RAG_DEFAULTS.maxQuestionChars) {
      setRagStatus(`问题过长（>${RAG_DEFAULTS.maxQuestionChars} 字符）`);
      return;
    }

    const style = getRagStyle();
    updateRagScope();
    setRagStatus("检索中…");
    setRagAnswerText("");
    setRagCitations([]);

    const retrieved = bm25.search(question, { allowedIds: currentPaperIds, topK: RAG_DEFAULTS.topK });
    const contexts = retrieved
      .map((x) => {
        const p = paperIndex.get(x.id);
        if (!p) return null;
        const tri = p.triple || {};
        return {
          id: p.id,
          score: x.score,
          title: p.title,
          venue: p.venue,
          year: p.year,
          url: p.url,
          keywords: p.keywords,
          method: tri.method,
          result: tri.result,
          contribution: tri.contribution,
          abstract: p.abstract,
        };
      })
      .filter(Boolean);

    if (!contexts.length) {
      setRagStatus("检索未命中（可尝试换关键词/切换主题范围）");
      return;
    }

    const apiKey = String(ragKey?.value || "").trim();
    if (!apiKey) {
      setRagStatus("请先填写 DeepSeek API Key（或改用带后端的本地运行模式）");
      return;
    }

    setRagStatus(`生成中…（${style === "cite" ? "精确引用型" : "综述型"}）`);

    const system =
      "You are a precise research assistant. " +
      "Answer based ONLY on the provided paper contexts. " +
      "If evidence is insufficient, say so. " +
      "Return JSON only.";

    const userObj = {
      question,
      style,
      scope: { papers_count: currentPaperIds.length, context: currentContextText },
      papers: contexts,
      output_schema:
        style === "cite"
          ? {
              answer_cn: "string",
              key_points: "list of short strings (optional)",
              citations: "list of paper ids used",
              notes: "optional string",
            }
          : {
              answer_cn: "string",
              key_points: "list of short strings (optional)",
              representative_papers: "list of paper ids (optional)",
              notes: "optional string",
            },
    };

    try {
      const out = await deepseekChatJson({
        apiKey,
        baseUrl: RAG_DEFAULTS.deepseekBaseUrl,
        model: RAG_DEFAULTS.model,
        system,
        userObj,
        maxTokens: 900,
        temperature: 0.2,
      });

      const answer = out?.answer_cn || "";
      const keyPoints = Array.isArray(out?.key_points) ? out.key_points : [];
      const citeIdsRaw = Array.isArray(out?.citations)
        ? out.citations
        : Array.isArray(out?.representative_papers)
          ? out.representative_papers
          : [];
      const citeIds = citeIdsRaw.map(String).filter((id) => paperIndex.has(id));

      const composed = [answer, keyPoints.length ? "\n\n要点：\n- " + keyPoints.map(String).join("\n- ") : ""].join("");
      setRagAnswerText(composed.trim());
      setRagCitations(citeIds);
      setRagStatus(`完成：检索命中 ${contexts.length} 篇`);
    } catch (e) {
      setRagStatus(`生成失败：${e?.message || e}`);
    }
  }

  if (ragSaveKey) ragSaveKey.addEventListener("click", () => {
    try {
      saveKey();
      setRagStatus("已保存到本地浏览器（localStorage）");
    } catch {
      setRagStatus("保存失败（浏览器限制）");
    }
  });
  if (ragClearKey) ragClearKey.addEventListener("click", () => {
    clearKey();
    setRagStatus("已清除本地 Key");
  });
  if (ragAsk) ragAsk.addEventListener("click", ragRun);
  if (ragQuestion) ragQuestion.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") ragRun();
  });

  if (ragPresets) {
    const list = Array.isArray(presets) ? presets : [];
    ragPresets.innerHTML = "";
    for (const it of list) {
      const btn = document.createElement("button");
      btn.className = "qa-preset-btn";
      btn.type = "button";
      btn.textContent = it.title || it.id || "preset";
      btn.title = it.question || "";
      btn.addEventListener("click", () => {
        if (ragQuestion) ragQuestion.value = it.question || "";
        ragRun();
      });
      ragPresets.appendChild(btn);
    }
  }

  loadSavedKey();
  updateRagScope();

  const toggle = document.getElementById("toggleDict");
  const toggleWrap = document.getElementById("dictToggleWrap");
  if (!dictMap) {
    // hide toggle if mapping not present
    if (toggleWrap) toggleWrap.style.display = "none";
  } else if (toggle) {
    toggle.checked = false;
    toggle.addEventListener("change", () => {
      viewMode = toggle.checked ? "dict" : "original";
      rerender();
    });
  }

  function focusText() {
    const parts = [];
    for (let col = 0; col < 3; col++) {
      const dim = columns[col];
      const ex = expanded[col];
      if (ex == null) continue;

      if (dim === "stage") {
        parts.push(`${dim}=${getStageLabel(stageMeta, "l1", String(ex))}`);
      } else {
        parts.push(`${dim}=${getMetaLabel(meta, dim, "l1", String(ex))}`);
      }
    }
    return parts.length ? `聚焦：${parts.join(" · ")}` : "全量";
  }

  // Footer counts (as the only global footer text)
  {
    const counts = new Map();
    for (const p of papers) {
      const k = venueKey(p.venue);
      counts.set(k, (counts.get(k) || 0) + 1);
    }
    if (footer) {
      const iclr = counts.get("ICLR") || 0;
      const icml = counts.get("ICML") || 0;
      const neurips = counts.get("NeurIPS") || 0;
      footer.textContent = `论文计数：ICLR ${iclr} · ICML ${icml} · NeurIPS ${neurips} · 合计 ${papers.length}`;
    }
  }

  function rerender() {
    const { nodes, links, nodePaperIds, linkPaperIds, nodeLabelMap } = buildSankey({
      papers,
      labels,
      meta,
      stageMeta,
      columns,
      expanded,
    });

    const nodeLabel = (name) => {
      const parsed = parseNodeName(name);
      if (!parsed || !parsed.dim) return nodeLabelMap.get(name) || name;
      // Dictionary view only applies to method/result L1
      if (viewMode === "dict" && parsed.kind === "l1" && (parsed.dim === "method" || parsed.dim === "result")) {
        const dl = getDictLabel(dictMap, parsed.dim, "l1", parsed.key);
        if (dl) return dl;
      }
      return nodeLabelMap.get(name) || name;
    };

    const option = {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "item",
        formatter: (p) => {
          if (p.dataType === "node") {
            const nm = nodeLabel(p.name);
            const cnt = nodePaperIds.get(p.name)?.size || 0;
            const info = parseNodeName(p.name);
            let extra = "";
            if (viewMode === "dict" && info.kind === "l1" && (info.dim === "method" || info.dim === "result")) {
              const desc = getDictDescription(dictMap, info.dim, "l1", info.key);
              if (desc) extra = `<br/><span class='muted'>字典说明：</span>${escapeHtml(desc)}`;
            }
            return `<b>${escapeHtml(nm)}</b><br/>dim: ${escapeHtml(info.dim)} · ${escapeHtml(info.kind)}<br/>papers: ${cnt}${extra}`;
          }
          if (p.dataType === "edge") {
            const s = nodeLabel(p.data.source);
            const t = nodeLabel(p.data.target);
            const k = `${p.data.source}->${p.data.target}`;
            const cnt = linkPaperIds.get(k)?.size || 0;
            return `<b>${escapeHtml(s)} → ${escapeHtml(t)}</b><br/>papers: ${cnt}`;
          }
          return "";
        },
      },
      series: [
        {
          type: "sankey",
          emphasis: { focus: "adjacency" },
          draggable: true,
          data: nodes,
          links: links,
          nodeAlign: "justify",
          nodeWidth: 16,
          nodeGap: 10,
          lineStyle: { color: "source", opacity: 0.28, curveness: 0.5 },
          itemStyle: { borderWidth: 1, borderColor: "rgba(17,24,39,0.18)" },
          label: {
            color: "rgba(17,24,39,0.86)",
            formatter: (p) => nodeLabel(p.name),
          },
        },
      ],
    };

    chart.setOption(option, true);

    // default details: focus mode should also narrow the right panel
    const focusPaperIds = papers
      .filter((p) => paperMatchesExpanded(labels, p.id, columns[0], expanded[0]))
      .filter((p) => paperMatchesExpanded(labels, p.id, columns[1], expanded[1]))
      .filter((p) => paperMatchesExpanded(labels, p.id, columns[2], expanded[2]))
      .map((p) => p.id);

    currentPaperIds = focusPaperIds;
    currentContextText = focusText();
    updateRagScope();
    const searchTerm = byId("search").value;
    renderDetails({ paperIndex, paperIds: currentPaperIds, searchTerm });

    const detailsHint = document.getElementById("detailsHint");
    if (detailsHint) {
      detailsHint.textContent = `${currentContextText} · 当前筛选论文：${currentPaperIds.length} 篇`;
    }

    chart.off("click");
    chart.on("click", (p) => {
      const searchTermNow = byId("search").value;
      if (p.dataType === "node") {
        const ids = [...(nodePaperIds.get(p.name) || [])];
        currentPaperIds = ids;
        currentContextText = `${focusText()} · 节点：${nodeLabel(p.name)}`;
        renderDetails({ paperIndex, paperIds: ids, searchTerm: searchTermNow });
      } else if (p.dataType === "edge") {
        const k = `${p.data.source}->${p.data.target}`;
        const ids = [...(linkPaperIds.get(k) || [])];
        currentPaperIds = ids;
        currentContextText = `${focusText()} · 连线：${nodeLabel(p.data.source)} → ${nodeLabel(p.data.target)}`;
        renderDetails({ paperIndex, paperIds: ids, searchTerm: searchTermNow });
      }

      const detailsHint = document.getElementById("detailsHint");
      if (detailsHint) {
        detailsHint.textContent = `${currentContextText} · 当前筛选论文：${currentPaperIds.length} 篇（展示前 ${Math.min(200, currentPaperIds.length)} 篇）`;
      }

      updateRagScope();
    });

    chart.off("dblclick");
    chart.on("dblclick", (p) => {
      if (p.dataType !== "node") return;
      const info = parseNodeName(p.name);
      // only toggle expand on L1 nodes
      if (info.kind !== "l1") return;
      const l1id = info.key;
      const col = info.col;
      expanded[col] = (expanded[col] === l1id ? null : l1id);
      rerender();
    });

    window.addEventListener("resize", () => chart.resize());
  }

  byId("reset").addEventListener("click", () => {
    expanded = { 0: null, 1: null, 2: null };
    rerender();
  });

  byId("collapse0").addEventListener("click", () => {
    expanded[0] = null;
    rerender();
  });
  byId("collapse1").addEventListener("click", () => {
    expanded[1] = null;
    rerender();
  });
  byId("collapse2").addEventListener("click", () => {
    expanded[2] = null;
    rerender();
  });

  byId("search").addEventListener("input", () => {
    renderDetails({ paperIndex, paperIds: currentPaperIds, searchTerm: byId("search").value });
  });

  hint.textContent = "渲染中…";
  rerender();
}

main().catch((e) => {
  console.error(e);
  const footer = document.getElementById("footerCounts");
  if (footer) footer.textContent = `加载失败：${e?.message || e}`;
});
