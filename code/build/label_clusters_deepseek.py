import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from deepseek_client import DeepSeekClient


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def cosine_topk(vecs: np.ndarray, k: int) -> list[list[int]]:
    # returns, for each row i, the top-k nearest neighbor indices (excluding itself)
    n = vecs.shape[0]
    sims = vecs @ vecs.T
    out: list[list[int]] = []
    for i in range(n):
        row = sims[i].copy()
        row[i] = -1.0
        idx = np.argsort(-row)[:k]
        out.append([int(x) for x in idx])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cluster-meta",
        default=r"F:\202507\AIH_SURVEY\data\cluster_meta.json",
    )
    parser.add_argument(
        "--papers",
        default=r"F:\202507\AIH_SURVEY\data\papers.json",
    )
    parser.add_argument(
        "--out",
        default=r"F:\202507\AIH_SURVEY\data\cluster_meta_llm.json",
        help="Output cluster meta with LLM labels",
    )
    parser.add_argument(
        "--model",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument("--dims", default="keywords,method,result,contribution")
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="limit clusters per dim (0=all)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate LLM name/description even if already present",
    )
    args = parser.parse_args()

    meta_path = Path(args.cluster_meta)
    papers_path = Path(args.papers)
    out_path = Path(args.out)

    meta = load_json(meta_path)
    papers = load_json(papers_path)
    paper_by_id = {p["id"]: p for p in papers}

    dims = [d.strip() for d in args.dims.split(",") if d.strip()]

    client = DeepSeekClient.from_env()
    st_model = SentenceTransformer(args.model)

    def system_prompt_for_dim(dim: str) -> str:
        base = (
            "You are a senior researcher. You must output ONLY valid JSON. "
            "Your job: give a human-friendly cluster name and a short explanation. "
            "Do not hallucinate. Use only the provided papers/examples. "
            "Name should be concise and suitable for a Sankey node label."
        )
        if dim == "method":
            return (
                base
                + " For method clusters: name should describe the common approach family (e.g., '检索增强医学推理', '医学影像分割训练策略'). "
                + "Avoid full sentences and avoid copying a single paper's method verbatim. Prefer 6-14 Chinese characters."
            )
        if dim == "result":
            return (
                base
                + " For result clusters: name should be a short outcome category (e.g., '推理准确率提升', '分割性能提升', '泛化与稳健性提升'). "
                + "Do NOT include specific numbers, dataset names, or long sentences. Prefer 6-14 Chinese characters."
            )
        return base

    out_meta = json.loads(json.dumps(meta))
    out_meta.setdefault("llm", {})
    out_meta["llm"].update({"provider": "deepseek", "model": "deepseek-chat"})

    for dim in dims:
        d = out_meta.get("dimensions", {}).get(dim)
        if not d:
            continue

        # Build list of L1 clusters (exclude -1)
        l1_items = []
        for k, v in d.get("l1", {}).items():
            try:
                cid = int(v.get("id"))
            except Exception:
                continue
            if cid == -1:
                continue
            l1_items.append((k, v))

        l1_items.sort(key=lambda kv: -int(kv[1].get("size", 0)))
        if args.limit:
            l1_items = l1_items[: args.limit]

        # Use existing label as text for embedding neighbor suggestion
        rep_texts = [kv[1].get("label", "") for kv in l1_items]
        if rep_texts:
            em = st_model.encode(rep_texts, batch_size=64, show_progress_bar=False)
            em = l2_normalize(np.asarray(em, dtype=np.float32))
            nn = cosine_topk(em, k=min(args.neighbors, max(0, len(rep_texts) - 1)))
        else:
            nn = []

        for idx, (cid_str, info) in enumerate(l1_items):
            if (not args.overwrite) and ("llm" in info and info["llm"].get("name_cn")):
                continue

            example_ids = info.get("example_paper_ids", [])[:8]
            examples = []
            for pid in example_ids:
                p = paper_by_id.get(pid)
                if not p:
                    continue
                examples.append(
                    {
                        "id": pid,
                        "title": p.get("title"),
                        "keywords": p.get("keywords"),
                        "triple": p.get("triple"),
                        "venue": p.get("venue"),
                        "year": p.get("year"),
                    }
                )

            neighbor_summaries = []
            if nn:
                for j in nn[idx][: args.neighbors]:
                    if j == idx:
                        continue
                    ncid, ninfo = l1_items[j]
                    neighbor_summaries.append(
                        {
                            "cluster_id": int(ninfo.get("id")),
                            "current_auto_label": ninfo.get("label"),
                            "size": ninfo.get("size"),
                        }
                    )

            user = {
                "task": "Name and explain this research-topic cluster.",
                "dimension": dim,
                "cluster": {
                    "cluster_id": int(info.get("id")),
                    "auto_label": info.get("label"),
                    "size": int(info.get("size", 0)),
                    "examples": examples,
                    "similar_clusters": neighbor_summaries,
                },
                "output_schema": {
                    "name_cn": "string (<=12 Chinese chars preferred)",
                    "description_cn": "1-2 Chinese sentences",
                    "tags": "list of 3-6 short strings (optional)",
                    "merge_suggestions": "list of {cluster_id:int, reason_cn:str} (optional)",
                },
            }

            resp = client.chat_json(
                system=system_prompt_for_dim(dim),
                user=json.dumps(user, ensure_ascii=False),
            )

            name_cn = resp.get("name_cn")
            desc_cn = resp.get("description_cn")
            tags = resp.get("tags")
            merges = resp.get("merge_suggestions")

            if not isinstance(name_cn, str) or not name_cn.strip():
                name_cn = info.get("label") or f"Cluster {info.get('id')}"
            if not isinstance(desc_cn, str):
                desc_cn = ""
            if not isinstance(tags, list):
                tags = []
            tags = [t for t in tags if isinstance(t, str) and t.strip()][:8]
            if not isinstance(merges, list):
                merges = []

            info.setdefault("llm", {})
            info["llm"].update(
                {
                    "name_cn": name_cn.strip(),
                    "description_cn": desc_cn.strip(),
                    "tags": tags,
                    "merge_suggestions": merges,
                }
            )

        # write back
        out_meta["dimensions"][dim]["l1"] = {k: v for k, v in d.get("l1", {}).items()}

    save_json(out_path, out_meta)
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
