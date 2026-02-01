import argparse
import csv
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup


USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
]


@dataclass
class CrawlConfig:
    term: str
    group: str
    venue: str
    year: int

    limit: int
    start_offset: int
    end_offset: Optional[int]
    max_results: Optional[int]

    request_delay_min_s: float
    request_delay_max_s: float
    page_chunk_size: int
    page_chunk_sleep_min_s: float
    page_chunk_sleep_max_s: float
    retry_max: int
    retry_backoff_base_s: float
    timeout_s: int

    proxy: Optional[str]
    proxy_pool: Optional[List[str]]
    proxy_index: int
    rotate_proxy_every_pages: int
    proxy_api_url: Optional[str]
    proxy_api_token: Optional[str]
    proxy_api_timeout_s: int

    deepseek_api_key: Optional[str]
    deepseek_base_url: str
    deepseek_model: str
    deepseek_enabled: bool

    out_prefix: Optional[str]
    finalize_json: bool


def _random_headers_json() -> Dict[str, str]:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json,text/*;q=0.99",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
        "Origin": "https://openreview.net",
        "Referer": "https://openreview.net/",
    }


def _random_headers_html() -> Dict[str, str]:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
        "Referer": "https://openreview.net/",
    }


def _sleep_polite(cfg: CrawlConfig) -> None:
    time.sleep(random.uniform(cfg.request_delay_min_s, cfg.request_delay_max_s))


def _build_session(cfg: CrawlConfig) -> requests.Session:
    session = requests.Session()
    if cfg.proxy:
        session.proxies.update({"http": cfg.proxy, "https": cfg.proxy})
    return session


def _apply_proxy(session: requests.Session, proxy_url: Optional[str]) -> None:
    session.proxies.clear()
    if proxy_url:
        session.proxies.update({"http": proxy_url, "https": proxy_url})


def _normalize_proxy_url(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return s
    if not re.match(r"^https?://", s, flags=re.IGNORECASE):
        s = "http://" + s
    return s


def _fetch_proxy_from_api(cfg: CrawlConfig) -> Optional[str]:
    if not cfg.proxy_api_url:
        return None

    headers: Dict[str, str] = {}
    if cfg.proxy_api_token:
        headers["Authorization"] = f"Bearer {cfg.proxy_api_token}"

    try:
        resp = requests.get(cfg.proxy_api_url, headers=headers, timeout=cfg.proxy_api_timeout_s)
        if resp.status_code != 200:
            print(f"❌ 代理 API HTTP {resp.status_code}: {cfg.proxy_api_url}")
            return None

        content_type = (resp.headers.get("Content-Type") or "").lower()
        text = (resp.text or "").strip()
        if not text:
            return None

        if "application/json" in content_type or (text.startswith("{") and text.endswith("}")):
            try:
                data = resp.json()
                if isinstance(data, dict):
                    proxy = data.get("proxy")
                    if not proxy and isinstance(data.get("data"), dict):
                        proxy = data["data"].get("proxy")
                    if isinstance(proxy, str) and proxy.strip():
                        return _normalize_proxy_url(proxy)
            except Exception:
                return None
            return None

        return _normalize_proxy_url(text)
    except requests.RequestException as e:
        print(f"🌐 代理 API 请求异常: {e}")
        return None


def _rotate_proxy(session: requests.Session, cfg: CrawlConfig) -> None:
    if cfg.proxy_api_url:
        new_proxy = _fetch_proxy_from_api(cfg)
        if new_proxy:
            cfg.proxy = new_proxy
            _apply_proxy(session, cfg.proxy)
            return

    if not cfg.proxy_pool:
        return
    if len(cfg.proxy_pool) == 1:
        cfg.proxy = _normalize_proxy_url(cfg.proxy_pool[0])
        _apply_proxy(session, cfg.proxy)
        return

    cfg.proxy_index = (cfg.proxy_index + 1) % len(cfg.proxy_pool)
    cfg.proxy = _normalize_proxy_url(cfg.proxy_pool[cfg.proxy_index])
    _apply_proxy(session, cfg.proxy)


def _compute_ratelimit_wait_s(resp: requests.Response) -> Optional[float]:
    """OpenReview 会返回：
    - Retry-After: 秒
    - ratelimit-reset: 秒（剩余）
    - x-ratelimit-reset: epoch 秒
    """
    retry_after = resp.headers.get("Retry-After")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass

    rl_reset = resp.headers.get("ratelimit-reset")
    if rl_reset:
        try:
            return max(0.0, float(rl_reset))
        except ValueError:
            pass

    x_rl_reset = resp.headers.get("x-ratelimit-reset")
    if x_rl_reset:
        try:
            reset_epoch = float(x_rl_reset)
            now = time.time()
            return max(0.0, reset_epoch - now)
        except ValueError:
            pass

    return None


def fetch_with_retry(
    session: requests.Session,
    url: str,
    cfg: CrawlConfig,
    params: Optional[Dict[str, Any]] = None,
    method: str = "GET",
    json_body: Optional[Dict[str, Any]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Optional[requests.Response]:
    for attempt in range(cfg.retry_max):
        try:
            headers = extra_headers or {}
            if method.upper() == "POST":
                resp = session.post(url, params=params, headers=headers, json=json_body, timeout=cfg.timeout_s)
            else:
                resp = session.get(url, params=params, headers=headers, timeout=cfg.timeout_s)

            if resp.status_code == 200:
                return resp

            if resp.status_code == 429:
                _rotate_proxy(session, cfg)
                if cfg.proxy:
                    print(f"🔁  429 触发切换代理：{cfg.proxy}")

                wait_s = _compute_ratelimit_wait_s(resp)
                if wait_s is None:
                    wait_s = (cfg.retry_backoff_base_s**attempt) * 5 + random.uniform(0, 3)
                else:
                    wait_s = max(wait_s, (cfg.retry_backoff_base_s**attempt) * 2 + random.uniform(0, 2))

                print(
                    f"⚠️  429 Too Many Requests: {url}，{wait_s:.1f}s 后重试 (attempt {attempt + 1}/{cfg.retry_max})"
                )
                time.sleep(wait_s)
                continue

            print(f"❌ HTTP {resp.status_code}: {url}")
            return None
        except requests.RequestException as e:
            wait_s = (cfg.retry_backoff_base_s**attempt) * 3 + random.uniform(0, 3)
            print(f"🌐 请求异常: {e}，{wait_s:.1f}s 后重试 (attempt {attempt + 1}/{cfg.retry_max})")
            time.sleep(wait_s)

    return None


def deepseek_summarize(title: str, abstract: str, cfg: CrawlConfig, session: requests.Session) -> Dict[str, Any]:
    if not cfg.deepseek_enabled:
        return {}
    if not cfg.deepseek_api_key:
        return {"error": "missing_api_key"}

    api_url = cfg.deepseek_base_url.rstrip("/") + "/chat/completions"

    system = "你是医学与机器学习交叉领域的论文速读助手。你必须严格按 JSON 输出，不要输出多余文本。"
    user = (
        "请基于以下论文标题与摘要生成中文总结，要求：\n"
        "1) 总字数 <= 50 词（尽量精炼）\n"
        "2) 提供 4~6 个关键词\n"
        "3) 给出三元组：方法-结果-贡献（每项尽量短）\n"
        "4) 严格输出 JSON，字段如下：\n"
        "{\n"
        "  \"summary_cn\": \"...\",\n"
        "  \"keywords\": [\"...\"],\n"
        "  \"triple\": {\"method\": \"...\", \"result\": \"...\", \"contribution\": \"...\"}\n"
        "}\n\n"
        f"标题：{title}\n\n摘要：{abstract}"
    )

    payload = {
        "model": cfg.deepseek_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": 300,
    }

    headers = {
        "Authorization": f"Bearer {cfg.deepseek_api_key}",
        "Content-Type": "application/json",
    }

    resp = fetch_with_retry(
        session,
        api_url,
        cfg,
        method="POST",
        json_body=payload,
        extra_headers=headers,
    )
    if not resp:
        return {"error": "llm_request_failed"}

    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", content)
            if m:
                return json.loads(m.group(0))
            return {"error": "bad_json", "raw": content[:500]}
    except Exception as e:
        return {"error": "exception", "raw": str(e)}


def _sanitize_token(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    return s or "term"


def build_out_prefix(cfg: CrawlConfig, end_offset: int) -> str:
    term_part = _sanitize_token(cfg.term)
    return cfg.out_prefix or f"openreview_iclr{cfg.year}_{term_part}_{cfg.start_offset}_{end_offset}"


class StreamWriter:
    def __init__(self, jsonl_path: str, csv_path: str):
        self.jsonl_path = jsonl_path
        self.csv_path = csv_path

        self._jsonl_fp = open(jsonl_path, "a", encoding="utf-8")
        self._csv_fp = open(csv_path, "a", encoding="utf-8-sig", newline="")

        self._csv_writer: Optional[csv.DictWriter] = None

    def _ensure_csv_writer(self) -> csv.DictWriter:
        if self._csv_writer:
            return self._csv_writer

        fieldnames = [
            "venue",
            "search_title",
            "full_title",
            "url",
            "year",
            "is_main_conference",
            "abstract_snippet",
            "abstract",
            "abstract_source_venue",
            "abstract_source_url",
            "openreview_id",
            "openreview_forum_id",
            "authors",
            "pdf_url",
            "summary_cn",
            "keywords",
            "triple_method",
            "triple_result",
            "triple_contribution",
            "llm_error",
        ]

        self._csv_writer = csv.DictWriter(self._csv_fp, fieldnames=fieldnames)
        if self._csv_fp.tell() == 0:
            self._csv_writer.writeheader()
            self._csv_fp.flush()
        return self._csv_writer

    def append(self, row: Dict[str, Any]) -> None:
        self._jsonl_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._jsonl_fp.flush()

        writer = self._ensure_csv_writer()
        triple = row.get("triple") or {}
        writer.writerow(
            {
                "venue": row.get("venue", ""),
                "search_title": row.get("search_title", ""),
                "full_title": row.get("full_title", ""),
                "url": row.get("url", ""),
                "year": row.get("year", ""),
                "is_main_conference": row.get("is_main_conference", False),
                "abstract_snippet": row.get("abstract_snippet", ""),
                "abstract": row.get("abstract", ""),
                "abstract_source_venue": row.get("abstract_source_venue", ""),
                "abstract_source_url": row.get("abstract_source_url", ""),
                "openreview_id": row.get("openreview_id", ""),
                "openreview_forum_id": row.get("openreview_forum_id", ""),
                "authors": "; ".join(row.get("authors") or []),
                "pdf_url": row.get("pdf_url", ""),
                "summary_cn": row.get("summary_cn", ""),
                "keywords": "; ".join(row.get("keywords") or []),
                "triple_method": triple.get("method", ""),
                "triple_result": triple.get("result", ""),
                "triple_contribution": triple.get("contribution", ""),
                "llm_error": row.get("llm_error", ""),
            }
        )
        self._csv_fp.flush()

    def close(self) -> None:
        self._jsonl_fp.close()
        self._csv_fp.close()


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def finalize_json(jsonl_path: str, json_path: str) -> None:
    rows = list(_iter_jsonl(jsonl_path))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def _extract_next_data_json(html: str) -> Optional[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.find("script", id="__NEXT_DATA__")
    if not tag or not tag.string:
        return None
    try:
        return json.loads(tag.string)
    except Exception:
        return None


def _walk_find_content(obj: Any) -> Optional[Dict[str, Any]]:
    """在 Next.js __NEXT_DATA__ 里尽量找到 submission 的 content。"""
    if isinstance(obj, dict):
        # 常见路径：... pageProps -> forumNote -> content
        if isinstance(obj.get("content"), dict) and ("title" in obj["content"] or "abstract" in obj["content"]):
            return obj["content"]
        for v in obj.values():
            found = _walk_find_content(v)
            if found:
                return found
    elif isinstance(obj, list):
        for it in obj:
            found = _walk_find_content(it)
            if found:
                return found
    return None


def extract_title_and_abstract_from_forum_html(html: str) -> Tuple[Optional[str], Optional[str]]:
    data = _extract_next_data_json(html)
    if data:
        content = _walk_find_content(data)
        if content:
            title = _unwrap_value(content.get("title"))
            abstract = _unwrap_value(content.get("abstract"))
            if isinstance(title, str) and title.strip():
                title = title.strip()
            else:
                title = None
            if isinstance(abstract, str) and abstract.strip():
                abstract = abstract.strip()
            else:
                abstract = None
            if title or abstract:
                return title, abstract

    # fallback：极简 HTML 解析（OpenReview 可能不保证）
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    title = title_tag.get_text(" ", strip=True) if title_tag else None
    if title:
        title = re.sub(r"\s*\|\s*OpenReview.*$", "", title).strip() or None
    return title, None


def openreview_search(
    session: requests.Session,
    cfg: CrawlConfig,
    offset: int,
) -> List[Dict[str, Any]]:
    url = "https://api2.openreview.net/notes/search"
    params = {
        "content": "all",
        "group": cfg.group,
        "limit": cfg.limit,
        "offset": offset,
        "source": "forum",
        "term": cfg.term,
        "type": "terms",
        "venue": cfg.venue,
    }

    resp = fetch_with_retry(session, url, cfg, params=params, extra_headers=_random_headers_json())
    if not resp:
        return []

    try:
        data = resp.json()
    except Exception:
        return []

    notes = data.get("notes")
    if not isinstance(notes, list):
        return []

    out: List[Dict[str, Any]] = []
    for n in notes:
        if not isinstance(n, dict):
            continue
        out.append(n)
    return out


def openreview_note_detail_api(
    session: requests.Session,
    cfg: CrawlConfig,
    note_id: str,
) -> Optional[Dict[str, Any]]:
    url = "https://api2.openreview.net/notes"
    params = {"id": note_id}
    resp = fetch_with_retry(session, url, cfg, params=params, extra_headers=_random_headers_json())
    if not resp:
        return None
    try:
        data = resp.json()
        notes = data.get("notes")
        if isinstance(notes, list) and notes and isinstance(notes[0], dict):
            return notes[0]
    except Exception:
        return None
    return None


def _unwrap_value(v: Any) -> Any:
    if isinstance(v, dict) and "value" in v:
        return v.get("value")
    return v


def _get_str(d: Dict[str, Any], key: str) -> Optional[str]:
    v = _unwrap_value(d.get(key))
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


def _get_list_str(d: Dict[str, Any], key: str) -> List[str]:
    v = _unwrap_value(d.get(key))
    if isinstance(v, list):
        out = [x.strip() for x in v if isinstance(x, str) and x.strip()]
        return out
    return []


def crawl(cfg: CrawlConfig) -> None:
    session = _build_session(cfg)

    # 估算 end_offset（用于输出前缀）
    if cfg.end_offset is not None:
        effective_end_offset = cfg.end_offset
    elif cfg.max_results is not None:
        effective_end_offset = cfg.start_offset + max(0, cfg.max_results - 1)
    else:
        # 不知道总量时先放大一点，最终会按实际停止
        effective_end_offset = cfg.start_offset

    out_prefix = build_out_prefix(cfg, effective_end_offset)
    out_jsonl = f"{out_prefix}.jsonl"
    out_csv = f"{out_prefix}.csv"
    out_json = f"{out_prefix}.json"

    print(
        f"🔍 OpenReview ICLR{cfg.year} | term={cfg.term} | group={cfg.group} | venue={cfg.venue} | limit={cfg.limit} | offset={cfg.start_offset} | 输出前缀: {out_prefix}"
    )

    writer = StreamWriter(out_jsonl, out_csv)

    seen_forums: set[str] = set()

    total_written = 0
    page_index = 0

    offset = cfg.start_offset
    while True:
        if cfg.end_offset is not None and offset > cfg.end_offset:
            break
        if cfg.max_results is not None and total_written >= cfg.max_results:
            break

        # 主动轮换：每 N 页切一次代理
        if cfg.rotate_proxy_every_pages and page_index > 0:
            if page_index % cfg.rotate_proxy_every_pages == 0:
                _rotate_proxy(session, cfg)
                if cfg.proxy:
                    print(f"🔁  主动切换代理（每 {cfg.rotate_proxy_every_pages} 页）：{cfg.proxy}")

        notes = openreview_search(session, cfg, offset)
        if not notes:
            print(f"✅ 搜索结果为空，停止：offset={offset}")
            break

        for n in notes:
            note_id = _get_str(n, "id")
            forum_id = _get_str(n, "forum") or note_id
            if not note_id or not forum_id:
                continue

            if forum_id in seen_forums:
                continue
            seen_forums.add(forum_id)

            if cfg.max_results is not None and total_written >= cfg.max_results:
                break

            forum_url = f"https://openreview.net/forum?id={forum_id}"
            pdf_url = f"https://openreview.net/pdf?id={forum_id}"

            # 先从搜索结果里拿到一个 snippet（有时 content.abstract 会被截断，有时完整）
            content = n.get("content") if isinstance(n.get("content"), dict) else {}
            search_title = _get_str(content, "title") or ""
            abstract_snippet = _get_str(content, "abstract") or ""
            search_authors = _get_list_str(content, "authors")
            if not search_authors:
                search_authors = _get_list_str(content, "authorids")

            pdf_from_content = _get_str(content, "pdf")
            if pdf_from_content:
                if pdf_from_content.startswith("/"):
                    pdf_url = "https://openreview.net" + pdf_from_content
                else:
                    pdf_url = pdf_from_content

            row: Dict[str, Any] = {
                "venue": "ICLR",
                "search_title": search_title,
                "full_title": search_title,
                "url": forum_url,
                "year": cfg.year,
                "is_main_conference": True,
                "abstract_snippet": abstract_snippet,
                "abstract": None,
                "abstract_source_venue": None,
                "abstract_source_url": None,
                "openreview_id": note_id,
                "openreview_forum_id": forum_id,
                "authors": search_authors,
                "pdf_url": pdf_url,
            }

            # 详情抓取：优先“点进去”（forum HTML），不行再用 notes?id API
            detail_title: Optional[str] = None
            detail_abs: Optional[str] = None
            detail_source: Optional[str] = None

            detail_html = fetch_with_retry(session, forum_url, cfg, extra_headers=_random_headers_html())
            if detail_html:
                t, a = extract_title_and_abstract_from_forum_html(detail_html.text)
                detail_title, detail_abs = t, a
                if detail_abs:
                    detail_source = "forum_html"

            if not detail_abs:
                # fallback: API 详情（通常稳定包含 title/abstract）
                detail_note = openreview_note_detail_api(session, cfg, note_id)
                if detail_note and isinstance(detail_note.get("content"), dict):
                    dc = detail_note["content"]
                    detail_title = detail_title or _get_str(dc, "title")
                    detail_abs = _get_str(dc, "abstract")
                    if detail_abs:
                        detail_source = "note_api"
                    if not row.get("authors"):
                        row["authors"] = _get_list_str(dc, "authors") or _get_list_str(dc, "authorids")

                    pdf_from_detail = _get_str(dc, "pdf")
                    if pdf_from_detail and not row.get("pdf_url"):
                        if pdf_from_detail.startswith("/"):
                            row["pdf_url"] = "https://openreview.net" + pdf_from_detail
                        else:
                            row["pdf_url"] = pdf_from_detail

            row["full_title"] = detail_title or row.get("search_title")
            row["abstract"] = detail_abs
            if detail_abs:
                row["abstract_source_venue"] = "OpenReviewForum" if detail_source == "forum_html" else "OpenReviewAPI"
                row["abstract_source_url"] = forum_url if detail_source == "forum_html" else "https://api2.openreview.net/notes?id=" + note_id
            else:
                row["llm_error"] = "abstract_not_found"

            if detail_abs:
                llm = deepseek_summarize(row["full_title"], detail_abs, cfg, session)
                if "error" in llm:
                    row["llm_error"] = llm.get("error")
                row.update({k: v for k, v in llm.items() if k in {"summary_cn", "keywords", "triple"}})

            writer.append(row)
            total_written += 1

            _sleep_polite(cfg)

        offset += cfg.limit
        page_index += 1

        # 每抓 N 页额外冷却
        if cfg.page_chunk_size > 0 and page_index % cfg.page_chunk_size == 0:
            cooldown_s = random.uniform(cfg.page_chunk_sleep_min_s, cfg.page_chunk_sleep_max_s)
            print(f"⏸️  已完成 {page_index} 页（约 {total_written} 条），额外冷却 {cooldown_s:.0f}s 后继续…")
            time.sleep(cooldown_s)

    writer.close()
    if cfg.finalize_json:
        finalize_json(out_jsonl, out_json)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OpenReview crawler for ICLR 2026 Conference notes")

    p.add_argument("--term", default="medical", help="检索 term（默认 medical）")
    p.add_argument("--group", default="ICLR.cc/2026/Conference", help="OpenReview group（默认 ICLR.cc/2026/Conference）")
    p.add_argument("--venue", default="ICLR 2026", help="OpenReview venue（默认 ICLR 2026）")
    p.add_argument("--year", type=int, default=2026, help="year 字段（默认 2026）")

    p.add_argument("--limit", type=int, default=25, help="每次拉取数量（默认 25）")
    p.add_argument("--start-offset", type=int, default=0)
    p.add_argument("--end-offset", type=int, default=None)
    p.add_argument("--max-results", type=int, default=200, help="最多输出多少条（默认 200，便于测试；设为 0/None 表示不限制）")

    p.add_argument("--delay-min", type=float, default=3.0)
    p.add_argument("--delay-max", type=float, default=6.0)
    p.add_argument("--page-chunk", type=int, default=2, help="每爬多少页触发一次额外冷却（默认 2 页）")
    p.add_argument("--chunk-sleep-min", type=float, default=20.0)
    p.add_argument("--chunk-sleep-max", type=float, default=40.0)
    p.add_argument("--retry-max", type=int, default=6)
    p.add_argument("--retry-backoff-base", type=float, default=2.0)
    p.add_argument("--timeout", type=int, default=20)

    p.add_argument("--proxy", default=None, help="代理地址，例如 http://127.0.0.1:7897 或 127.0.0.1:7897")
    p.add_argument("--proxy-list", default=None, help="多个代理（逗号分隔），例如 host1:port,host2:port")
    p.add_argument("--proxy-user", default=None)
    p.add_argument("--proxy-pass", default=None)
    p.add_argument("--rotate-proxy-every-pages", type=int, default=5)

    p.add_argument("--proxy-api-url", default=None)
    p.add_argument("--proxy-api-token", default=None)
    p.add_argument("--proxy-api-timeout", type=int, default=10)

    p.add_argument("--deepseek-api-key", default=None)
    p.add_argument("--deepseek-base-url", default=None)
    p.add_argument("--deepseek-model", default="deepseek-chat")
    p.add_argument("--no-llm", action="store_true")

    p.add_argument("--out-prefix", default=None)
    p.add_argument("--no-finalize-json", action="store_true")

    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    deepseek_api_key = args.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY")
    deepseek_base_url = args.deepseek_base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    proxy_api_token = args.proxy_api_token or os.getenv("PROXY_API_TOKEN")

    def _build_proxy_url(raw: str, user: Optional[str], password: Optional[str]) -> str:
        if not (user and password):
            return raw
        import urllib.parse

        user_enc = urllib.parse.quote(user, safe="")
        pass_enc = urllib.parse.quote(password, safe="")
        m = re.match(r"^(?:http[s]?://)?([^@/]+)(?::(\d+))?$", raw)
        if m:
            host = m.group(1)
            port = m.group(2) or ""
            hostport = f"{host}:{port}" if port else host
            return f"http://{user_enc}:{pass_enc}@{hostport}"
        return f"http://{user_enc}:{pass_enc}@{raw.lstrip('http://').lstrip('https://')}"

    proxy_pool: Optional[List[str]] = None
    if args.proxy_list:
        raw_list = [p.strip() for p in args.proxy_list.split(",") if p.strip()]
        if args.proxy_user and args.proxy_pass:
            proxy_pool = [_build_proxy_url(p, args.proxy_user, args.proxy_pass) for p in raw_list]
        else:
            proxy_pool = raw_list
    elif args.proxy_user and args.proxy_pass:
        if not args.proxy:
            raise ValueError("--proxy-user/--proxy-pass 需要配合 --proxy 指定主机:端口")
        proxy_pool = [_build_proxy_url(args.proxy, args.proxy_user, args.proxy_pass)]
    elif args.proxy:
        proxy_pool = [args.proxy]

    proxy_url = _normalize_proxy_url(proxy_pool[0]) if proxy_pool else None

    max_results: Optional[int]
    if args.max_results is None or int(args.max_results) <= 0:
        max_results = None
    else:
        max_results = int(args.max_results)

    cfg = CrawlConfig(
        term=args.term,
        group=args.group,
        venue=args.venue,
        year=int(args.year),
        limit=max(1, int(args.limit)),
        start_offset=max(0, int(args.start_offset)),
        end_offset=args.end_offset,
        max_results=max_results,
        request_delay_min_s=args.delay_min,
        request_delay_max_s=args.delay_max,
        page_chunk_size=args.page_chunk,
        page_chunk_sleep_min_s=args.chunk_sleep_min,
        page_chunk_sleep_max_s=args.chunk_sleep_max,
        retry_max=args.retry_max,
        retry_backoff_base_s=args.retry_backoff_base,
        timeout_s=args.timeout,
        proxy=proxy_url,
        proxy_pool=proxy_pool,
        proxy_index=0,
        rotate_proxy_every_pages=max(0, int(args.rotate_proxy_every_pages)),
        proxy_api_url=args.proxy_api_url,
        proxy_api_token=proxy_api_token,
        proxy_api_timeout_s=args.proxy_api_timeout,
        deepseek_api_key=deepseek_api_key,
        deepseek_base_url=deepseek_base_url,
        deepseek_model=args.deepseek_model,
        deepseek_enabled=not args.no_llm,
        out_prefix=args.out_prefix,
        finalize_json=not args.no_finalize_json,
    )

    if cfg.proxy_pool:
        cfg.proxy_pool = [_normalize_proxy_url(p) for p in cfg.proxy_pool if p and str(p).strip()]
        if cfg.proxy_pool:
            cfg.proxy = cfg.proxy or cfg.proxy_pool[0]

    crawl(cfg)


if __name__ == "__main__":
    main()
