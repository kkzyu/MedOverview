import argparse
import csv
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
]


@dataclass(frozen=True)
class VenueSpec:
    key: str  # icml / iclr / neurips
    display: str
    base_url: str  # icml.cc / iclr.cc / neurips.cc
    search_url: str
    proceedings_host: Optional[str]  # proceedings.iclr.cc / proceedings.neurips.cc


VENUES: Dict[str, VenueSpec] = {
    "icml": VenueSpec(
        key="icml",
        display="ICML",
        base_url="https://icml.cc",
        search_url="https://icml.cc/search",
        proceedings_host=None,
    ),
    "iclr": VenueSpec(
        key="iclr",
        display="ICLR",
        base_url="https://iclr.cc",
        search_url="https://iclr.cc/search",
        proceedings_host="proceedings.iclr.cc",
    ),
    "neurips": VenueSpec(
        key="neurips",
        display="NeurIPS",
        base_url="https://neurips.cc",
        search_url="https://neurips.cc/search",
        proceedings_host="proceedings.neurips.cc",
    ),
}


@dataclass
class CrawlConfig:
    venue: VenueSpec
    query: str
    years: Tuple[int, ...]
    start_page: int
    end_page: Optional[int]
    max_pages: Optional[int]

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

    deepseek_api_key: Optional[str]
    deepseek_base_url: str
    deepseek_model: str
    deepseek_enabled: bool

    # output
    out_prefix: Optional[str]
    finalize_json: bool


def _random_headers(cfg: CrawlConfig) -> Dict[str, str]:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": cfg.venue.base_url + "/",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
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


def _rotate_proxy(session: requests.Session, cfg: CrawlConfig) -> None:
    if not cfg.proxy_pool or len(cfg.proxy_pool) < 2:
        return
    cfg.proxy_index = (cfg.proxy_index + 1) % len(cfg.proxy_pool)
    cfg.proxy = cfg.proxy_pool[cfg.proxy_index]
    _apply_proxy(session, cfg.proxy)
    print(f"🔁  429 触发切换代理：{cfg.proxy}")


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
            headers = _random_headers(cfg)
            if extra_headers:
                headers.update(extra_headers)

            if method.upper() == "POST":
                resp = session.post(url, params=params, headers=headers, json=json_body, timeout=cfg.timeout_s)
            else:
                resp = session.get(url, params=params, headers=headers, timeout=cfg.timeout_s)

            if resp.status_code == 200:
                return resp

            if resp.status_code == 429:
                _rotate_proxy(session, cfg)
                retry_after = resp.headers.get("Retry-After")
                retry_after_s: Optional[float] = None
                if retry_after:
                    try:
                        retry_after_s = float(retry_after)
                    except ValueError:
                        retry_after_s = None

                # 429 往往需要更长等待：指数退避 + 抖动，并尊重 Retry-After
                wait_s = (cfg.retry_backoff_base_s ** attempt) * 5 + random.uniform(0, 3)
                if retry_after_s is not None:
                    wait_s = max(wait_s, retry_after_s)
                print(
                    f"⚠️  429 Too Many Requests: {url}，{wait_s:.1f}s 后重试 (attempt {attempt + 1}/{cfg.retry_max})"
                )
                time.sleep(wait_s)
                continue

            print(f"❌ HTTP {resp.status_code}: {url}")
            return None
        except requests.RequestException as e:
            # 代理断连/远端直接断开时，通常也需要更长间隔再试
            wait_s = (cfg.retry_backoff_base_s ** attempt) * 3 + random.uniform(0, 3)
            print(f"🌐 请求异常: {e}，{wait_s:.1f}s 后重试 (attempt {attempt + 1}/{cfg.retry_max})")
            time.sleep(wait_s)

    return None


def parse_total_pages(html: str) -> int:
    m = re.search(r"Page\s+\d+\s+of\s+(\d+)", html)
    if not m:
        return 1
    try:
        return int(m.group(1))
    except ValueError:
        return 1


def normalize_url(cfg: CrawlConfig, href: str) -> str:
    if href.startswith("/"):
        return urljoin(cfg.venue.base_url, href)
    return href


def _year_from_virtual_url(url: str) -> Optional[int]:
    m = re.search(r"/virtual/(\d{4})/", url)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _year_from_proceedings_url(url: str, proceedings_host: str) -> Optional[int]:
    m = re.search(rf"{re.escape(proceedings_host)}/paper_files/paper/(\d{{4}})/", url)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def extract_year(cfg: CrawlConfig, url: str) -> Optional[int]:
    y = _year_from_virtual_url(url)
    if y:
        return y
    if cfg.venue.proceedings_host:
        return _year_from_proceedings_url(url, cfg.venue.proceedings_host)
    return None


def is_main_conference_url(cfg: CrawlConfig, url: str) -> bool:
    year = extract_year(cfg, url)
    if year not in cfg.years:
        return False

    # virtual: https://<venue>.cc/virtual/2024/poster/xxxxx OR /virtual/2024/xxxxx
    if re.match(rf"^{re.escape(cfg.venue.base_url)}/virtual/(\d{{4}})/(poster/\d+|\d+)/?$", url.strip()):
        return True

    # proceedings only for ICLR/NeurIPS
    if cfg.venue.proceedings_host:
        # ICLR: ...-Abstract-Conference.html
        # NeurIPS: ...-Abstract-<Track>.html
        if re.match(
            rf"^https://{re.escape(cfg.venue.proceedings_host)}/paper_files/paper/(\d{{4}})/hash/.+-Abstract-.+\.html$",
            url.strip(),
        ):
            return True

    return False


def parse_search_page(cfg: CrawlConfig, html: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    items: List[Dict[str, Any]] = []

    for res in soup.find_all("div", class_="gsc-result"):
        link = res.find("a", class_="gs-title")
        if not link:
            title_div = res.find("div", class_="gs-title")
            link = title_div.find("a") if title_div else None
        if not link or not link.get("href"):
            continue

        title = link.get_text(strip=True)
        url = normalize_url(cfg, link["href"])

        snippet = ""
        snippet_tag = res.find("div", class_="gs-snippet")
        if snippet_tag:
            snippet = snippet_tag.get_text(" ", strip=True)

        items.append(
            {
                "venue": cfg.venue.display,
                "search_title": title,
                "url": url,
                "year": extract_year(cfg, url),
                "abstract_snippet": snippet,
            }
        )

    return items


def _extract_abstract_by_heading(soup: BeautifulSoup) -> Optional[str]:
    # Virtual sites: <h3>Abstract</h3> then <p>
    h3 = soup.find(lambda tag: tag.name == "h3" and "Abstract" in tag.get_text())
    if h3:
        p = h3.find_next("p")
        if p:
            return p.get_text(" ", strip=True)

    # Proceedings: often <h2>Abstract</h2> / <h3>Abstract</h3>
    for header_name in ("h2", "h3", "h4"):
        h = soup.find(lambda tag: tag.name == header_name and tag.get_text(strip=True).lower() == "abstract")
        if h:
            p = h.find_next("p")
            if p:
                return p.get_text(" ", strip=True)

    return None


def extract_title_and_abstract(cfg: CrawlConfig, url: str, html: str) -> Tuple[Optional[str], Optional[str]]:
    soup = BeautifulSoup(html, "html.parser")

    full_title: Optional[str] = None

    h1 = soup.find("h1")
    if h1:
        full_title = h1.get_text(" ", strip=True)

    if not full_title:
        title_like = soup.find(
            lambda tag: tag.name in {"h1", "h2", "h3"}
            and tag.get("class")
            and any("title" in str(c).lower() for c in tag.get("class"))
        )
        if title_like:
            full_title = title_like.get_text(" ", strip=True)

    if not full_title:
        title_tag = soup.find("title")
        if title_tag:
            full_title = title_tag.get_text(" ", strip=True)
            full_title = re.sub(r"\s*\|\s*(ICML|ICLR|NeurIPS).*$", "", full_title).strip()

    abstract: Optional[str] = _extract_abstract_by_heading(soup)

    if not abstract:
        meta = soup.find("meta", attrs={"name": "citation_abstract"})
        if meta and meta.get("content"):
            abstract = meta["content"].strip()

    if not abstract:
        for p in soup.find_all("p"):
            txt = p.get_text(" ", strip=True)
            if "abstract" in txt.lower() and len(txt) > 200:
                abstract = re.sub(r"^Abstract\s*:?\s*", "", txt, flags=re.IGNORECASE).strip()
                break

    return full_title, abstract


def deepseek_summarize(
    title: str,
    abstract: str,
    cfg: CrawlConfig,
    session: requests.Session,
) -> Dict[str, Any]:
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
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    return s or "query"


def build_out_prefix(cfg: CrawlConfig, effective_end_page: int) -> str:
    years_part = "_".join(str(y) for y in cfg.years)
    query_part = _sanitize_token(cfg.query)
    return cfg.out_prefix or f"{cfg.venue.key}_{query_part}_{years_part}_{cfg.start_page}_{effective_end_page}"


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
            "summary_cn",
            "keywords",
            "triple_method",
            "triple_result",
            "triple_contribution",
            "llm_error",
        ]

        self._csv_writer = csv.DictWriter(self._csv_fp, fieldnames=fieldnames)
        # 如果文件为空，写 header
        if self._csv_fp.tell() == 0:
            self._csv_writer.writeheader()
            self._csv_fp.flush()
        return self._csv_writer

    def append(self, row: Dict[str, Any]) -> None:
        # JSONL
        self._jsonl_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._jsonl_fp.flush()

        # CSV
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


def crawl(cfg: CrawlConfig) -> None:
    session = _build_session(cfg)

    first = fetch_with_retry(
        session,
        cfg.venue.search_url,
        cfg,
        params={"q": cfg.query, "page": cfg.start_page},
    )
    if not first:
        raise RuntimeError("无法抓取搜索第一页，请检查网络/代理/限速")

    total_pages = parse_total_pages(first.text)
    if cfg.end_page is not None:
        effective_end = min(cfg.end_page, total_pages)
    else:
        effective_end = total_pages

    if cfg.max_pages is not None:
        effective_end = min(effective_end, cfg.start_page + cfg.max_pages - 1)

    out_prefix = build_out_prefix(cfg, effective_end)
    out_jsonl = f"{out_prefix}.jsonl"
    out_csv = f"{out_prefix}.csv"
    out_json = f"{out_prefix}.json"

    print(
        f"🔍 {cfg.venue.display} 搜索词: {cfg.query} | 年份: {cfg.years} | 页数: {cfg.start_page}~{effective_end} | 输出前缀: {out_prefix}"
    )

    writer = StreamWriter(out_jsonl, out_csv)

    seen_urls: set[str] = set()

    def handle_search_html(html: str) -> List[Dict[str, Any]]:
        items = parse_search_page(cfg, html)
        new_items: List[Dict[str, Any]] = []
        for it in items:
            url = it["url"]
            if url in seen_urls:
                continue
            seen_urls.add(url)
            it["is_main_conference"] = is_main_conference_url(cfg, url)
            new_items.append(it)
        return new_items

    # page loop: parse -> for each item, if main then detail+llm -> write immediately
    for page in range(cfg.start_page, effective_end + 1):
        if page == cfg.start_page:
            resp = first
        else:
            resp = fetch_with_retry(session, cfg.venue.search_url, cfg, params={"q": cfg.query, "page": page})

        if not resp:
            print(f"⚠️  跳过搜索页: {page}")
            _sleep_polite(cfg)
            continue

        page_items = handle_search_html(resp.text)

        for it in page_items:
            url = it["url"]
            it["abstract_source_venue"] = None
            it["abstract_source_url"] = None

            if it.get("is_main_conference"):
                detail = fetch_with_retry(session, url, cfg)
                if detail:
                    full_title, abstract = extract_title_and_abstract(cfg, url, detail.text)
                    it["full_title"] = full_title or it.get("search_title")
                    it["abstract"] = abstract
                    it["abstract_source_venue"] = cfg.venue.display
                    it["abstract_source_url"] = url

                    if abstract:
                        llm = deepseek_summarize(it["full_title"], abstract, cfg, session)
                        if "error" in llm:
                            it["llm_error"] = llm.get("error")
                        it.update({k: v for k, v in llm.items() if k in {"summary_cn", "keywords", "triple"}})
                    else:
                        it["llm_error"] = "abstract_not_found"
                else:
                    it["full_title"] = it.get("search_title")
                    it["abstract"] = None
                    it["llm_error"] = "detail_fetch_failed"

                writer.append(it)
                _sleep_polite(cfg)
            else:
                # 非主会：也写一条，方便形成完整列表
                it["full_title"] = it.get("search_title")
                it["abstract"] = None
                writer.append(it)

        _sleep_polite(cfg)

        # 每爬取 N 页，额外长时间冷却，降低被封概率
        if cfg.page_chunk_size > 0:
            pages_done = page - cfg.start_page + 1
            if pages_done % cfg.page_chunk_size == 0 and page < effective_end:
                cooldown_s = random.uniform(cfg.page_chunk_sleep_min_s, cfg.page_chunk_sleep_max_s)
                print(f"⏸️  已完成 {pages_done} 页，额外冷却 {cooldown_s:.0f}s 后继续…")
                time.sleep(cooldown_s)

    writer.close()

    if cfg.finalize_json:
        finalize_json(out_jsonl, out_json)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified crawler for ICML/ICLR/NeurIPS medical-related papers")
    p.add_argument("--venue", required=True, choices=sorted(VENUES.keys()), help="icml | iclr | neurips")
    p.add_argument("--query", default=None, help="搜索词（默认：icml=Med, iclr/neurips=medical）")
    p.add_argument("--years", default="2024,2025", help="目标年份，默认 2024,2025")
    p.add_argument("--start-page", type=int, default=50)
    p.add_argument("--end-page", type=int, default=None)
    p.add_argument("--max-pages", type=int, default=100, help="从 start-page 起最多抓多少页（便于测试）")

    p.add_argument("--delay-min", type=float, default=4.0)
    p.add_argument("--delay-max", type=float, default=8.0)
    p.add_argument("--page-chunk", type=int, default=2, help="每爬多少页触发一次额外冷却（默认 3 页）")
    p.add_argument("--chunk-sleep-min", type=float, default=30.0, help="额外冷却最小秒数（默认 30s）")
    p.add_argument("--chunk-sleep-max", type=float, default=60.0, help="额外冷却最大秒数（默认 60s）")
    p.add_argument("--retry-max", type=int, default=5)
    p.add_argument("--retry-backoff-base", type=float, default=2.0)
    p.add_argument("--timeout", type=int, default=20)
    p.add_argument("--proxy", default=None, help="代理地址，例如 http://127.0.0.1:7897 或 http://user:pass@host:port")
    p.add_argument("--proxy-list", default=None, help="多个代理（逗号分隔），例如 host1:port,host2:port")
    p.add_argument("--proxy-user", default=None, help="代理用户名（可选，和 --proxy-pass 一起用）")
    p.add_argument("--proxy-pass", default=None, help="代理密码（可选，和 --proxy-user 一起用）")

    p.add_argument(
        "--deepseek-api-key",
        default=None,
        help="DeepSeek API Key（也可用环境变量 DEEPSEEK_API_KEY；不要把 key 写进代码/仓库）",
    )
    p.add_argument("--deepseek-base-url", default=None, help="DeepSeek Base URL（默认 https://api.deepseek.com/v1）")
    p.add_argument("--deepseek-model", default="deepseek-chat", help="模型名（如 deepseek-v3.2）")
    p.add_argument("--no-llm", action="store_true", help="不调用 LLM，仅爬取")

    p.add_argument(
        "--out-prefix",
        default=None,
        help="输出文件前缀（默认自动生成：{venue}_{query}_{years}_{start}_{end}）",
    )
    p.add_argument("--no-finalize-json", action="store_true", help="不生成最终 .json（仅保留 .jsonl/.csv）")

    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    venue = VENUES[args.venue]

    default_query = "Med" if venue.key == "icml" else "medical"
    query = args.query or default_query

    years = tuple(int(x.strip()) for x in args.years.split(",") if x.strip())

    deepseek_api_key = args.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY")
    deepseek_base_url = args.deepseek_base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")


    # 动态拼接代理URL（优先级：user+pass > proxy/proxy-list）
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

    proxy_url = proxy_pool[0] if proxy_pool else None

    cfg = CrawlConfig(
        venue=venue,
        query=query,
        years=years,
        start_page=args.start_page,
        end_page=args.end_page,
        max_pages=args.max_pages,
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
        deepseek_api_key=deepseek_api_key,
        deepseek_base_url=deepseek_base_url,
        deepseek_model=args.deepseek_model,
        deepseek_enabled=not args.no_llm,
        out_prefix=args.out_prefix,
        finalize_json=not args.no_finalize_json,
    )

    crawl(cfg)


if __name__ == "__main__":
    main()
