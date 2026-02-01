import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import hdbscan
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency 'hdbscan'. Install with: pip install hdbscan"
    ) from e


def stable_id(url: str | None, title: str | None) -> str:
    base = (url or "") + "|" + (title or "")
    return hashlib.md5(base.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]


def safe_str(x: Any) -> str:
    return x if isinstance(x, str) else ""


def norm_kw(s: str) -> str:
    return " ".join(s.strip().lower().split())


@dataclass
class ClusterResult:
    labels: list[int]
    id_to_members: dict[int, list[int]]


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def hdbscan_cluster(emb: np.ndarray, min_cluster_size: int, min_samples: int) -> ClusterResult:
    if len(emb) == 0:
        return ClusterResult(labels=[], id_to_members={})

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(emb).tolist()

    id_to_members: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        id_to_members[lab].append(i)

    return ClusterResult(labels=labels, id_to_members=dict(id_to_members))


def assign_noise_to_nearest_cluster(emb_normed: np.ndarray, labels: list[int]) -> list[int]:
    # Replace -1 labels by nearest (cosine) non-noise cluster centroid.
    n = len(labels)
    if n == 0:
        return labels

    cluster_to_members: dict[int, list[int]] = defaultdict(list)
    noise = []
    for i, lab in enumerate(labels):
        if lab == -1:
            noise.append(i)
        else:
            cluster_to_members[lab].append(i)

    if not noise or not cluster_to_members:
        return labels

    cluster_ids = sorted(cluster_to_members.keys())
    centroids = []
    for cid in cluster_ids:
        m = cluster_to_members[cid]
        c = emb_normed[m].mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)
        centroids.append(c)
    centroids = np.stack(centroids, axis=0)  # [C, D]

    for i in noise:
        sims = centroids @ emb_normed[i]
        best = int(np.argmax(sims))
        labels[i] = int(cluster_ids[best])

    return labels


def pick_representative_text(
    member_idxs: list[int],
    texts: list[str],
    emb_normed: np.ndarray,
) -> str:
    if not member_idxs:
        return ""
    if len(member_idxs) == 1:
        return texts[member_idxs[0]][:80]

    sub = emb_normed[member_idxs]
    centroid = sub.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
    sims = sub @ centroid
    best_local = int(np.argmax(sims))
    return texts[member_idxs[best_local]][:80]


def top_keywords_for_cluster(member_idxs: list[int], paper_keywords: list[list[str]], topn: int = 4) -> list[str]:
    stop_norm = {
        "machine learning",
        "deep learning",
        "representation learning",
        "foundation model",
        "large language model",
        "llm",
        "医学",
        "医疗",
        "医学人工智能",
        "医疗ai",
        "人工智能",
        "机器学习",
        "深度学习",
        "表示学习",
        "基础模型",
        "大语言模型",
        "多模态",
        "transformer",
    }
    freq_norm: Counter[str] = Counter()
    norm_to_best: dict[str, str] = {}
    for i in member_idxs:
        for kw in paper_keywords[i]:
            if not isinstance(kw, str):
                continue
            n = norm_kw(kw)
            if not n:
                continue
            if n in stop_norm:
                continue
            freq_norm[n] += 1
            if n not in norm_to_best or len(kw) < len(norm_to_best[n]):
                norm_to_best[n] = kw

    tops = [n for n, _ in freq_norm.most_common(topn)]
    return [norm_to_best[n] for n in tops if n in norm_to_best]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_existing_labels(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def build_two_level_labels(
    texts: list[str],
    emb_normed: np.ndarray,
    min_cluster_size_l1: int,
    min_samples_l1: int,
    min_cluster_size_l2: int,
    min_samples_l2: int,
) -> tuple[list[int], list[int]]:
    # L1
    l1 = hdbscan_cluster(emb_normed, min_cluster_size_l1, min_samples_l1)
    l1_labels = assign_noise_to_nearest_cluster(emb_normed, l1.labels)

    # L2: only within non-noise L1 clusters
    l2_labels = [-1] * len(texts)

    for l1_id, members in l1.id_to_members.items():
        if l1_id == -1:
            continue
        if len(members) < max(min_cluster_size_l2, 3):
            for i in members:
                l2_labels[i] = 0
            continue

        sub_emb = emb_normed[members]
        sub = hdbscan_cluster(sub_emb, min_cluster_size_l2, min_samples_l2)

        # assign L2 noise to nearest L2 centroid within this L1
        sub_labels = sub.labels
        if any(x == -1 for x in sub_labels) and any(x != -1 for x in sub_labels):
            sub_labels = assign_noise_to_nearest_cluster(sub_emb, sub_labels)

        # map sub labels to stable positive ids per L1
        sub_map: dict[int, int] = {}
        next_id = 0
        for local_i, sub_lab in enumerate(sub_labels):
            if sub_lab == -1:
                continue
            if sub_lab not in sub_map:
                sub_map[sub_lab] = next_id
                next_id += 1

        for local_i, global_i in enumerate(members):
            sub_lab = sub_labels[local_i]
            if sub_lab == -1:
                l2_labels[global_i] = -1
            else:
                l2_labels[global_i] = sub_map[sub_lab]

    return l1_labels, l2_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        default=r"F:\202507\AIH_SURVEY\medical_2024_2025.jsonl",
        help="Path to medical_2024_2025.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        default=r"F:\202507\AIH_SURVEY\data",
        help="Output directory",
    )
    parser.add_argument(
        "--model",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument("--years", default="2024,2025")
    parser.add_argument("--min-cluster-size-l1", type=int, default=10)
    parser.add_argument("--min-samples-l1", type=int, default=2)
    parser.add_argument("--min-cluster-size-l2", type=int, default=5)
    parser.add_argument("--min-samples-l2", type=int, default=2)
    args = parser.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    existing_labels = load_existing_labels(out_dir / "labels.json")

    year_set = {int(x.strip()) for x in args.years.split(",") if x.strip()}

    # 1) load & filter
    papers: list[dict[str, Any]] = []
    paper_keywords: list[list[str]] = []
    kw_texts: list[str] = []
    method_texts: list[str] = []
    result_texts: list[str] = []
    contrib_texts: list[str] = []

    for obj in iter_jsonl(src):
        year = obj.get("year")
        if year not in year_set:
            continue
        if obj.get("is_main_conference") is not True:
            continue
        full_title = safe_str(obj.get("full_title"))
        abstract = safe_str(obj.get("abstract"))
        if not full_title or not abstract:
            continue

        kws = obj.get("keywords")
        if not isinstance(kws, list) or not kws:
            continue
        kws = [k for k in kws if isinstance(k, str) and k.strip()]
        if not kws:
            continue

        triple = obj.get("triple") if isinstance(obj.get("triple"), dict) else {}
        method = safe_str(triple.get("method"))
        result = safe_str(triple.get("result"))
        contribution = safe_str(triple.get("contribution"))

        pid = stable_id(safe_str(obj.get("url")), full_title)
        papers.append(
            {
                "id": pid,
                "venue": obj.get("venue"),
                "year": year,
                "title": full_title,
                "url": obj.get("url"),
                "abstract": abstract,
                "summary_cn": obj.get("summary_cn"),
                "keywords": kws,
                "triple": {
                    "method": method,
                    "result": result,
                    "contribution": contribution,
                },
            }
        )

        paper_keywords.append(kws)
        kw_texts.append("；".join(kws))
        method_texts.append(method or "(empty method)")
        result_texts.append(result or "(empty result)")
        contrib_texts.append(contribution or "(empty contribution)")

    if not papers:
        raise SystemExit("No papers after filtering. Check years / is_main_conference / abstract fields.")

    # 2) embeddings
    model = SentenceTransformer(args.model)

    def embed(texts: list[str]) -> np.ndarray:
        em = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=False)
        em = np.asarray(em, dtype=np.float32)
        return l2_normalize(em)

    kw_emb = embed(kw_texts)
    method_emb = embed(method_texts)
    result_emb = embed(result_texts)
    contrib_emb = embed(contrib_texts)

    # 3) L1/L2 labels
    kw_l1, kw_l2 = build_two_level_labels(
        kw_texts,
        kw_emb,
        args.min_cluster_size_l1,
        args.min_samples_l1,
        args.min_cluster_size_l2,
        args.min_samples_l2,
    )
    m_l1, m_l2 = build_two_level_labels(
        method_texts,
        method_emb,
        args.min_cluster_size_l1,
        args.min_samples_l1,
        args.min_cluster_size_l2,
        args.min_samples_l2,
    )
    r_l1, r_l2 = build_two_level_labels(
        result_texts,
        result_emb,
        args.min_cluster_size_l1,
        args.min_samples_l1,
        args.min_cluster_size_l2,
        args.min_samples_l2,
    )
    c_l1, c_l2 = build_two_level_labels(
        contrib_texts,
        contrib_emb,
        args.min_cluster_size_l1,
        args.min_samples_l1,
        args.min_cluster_size_l2,
        args.min_samples_l2,
    )

    # 4) cluster metadata
    cluster_meta: dict[str, Any] = {
        "model": args.model,
        "years": sorted(year_set),
        "min_cluster_size_l1": args.min_cluster_size_l1,
        "min_samples_l1": args.min_samples_l1,
        "min_cluster_size_l2": args.min_cluster_size_l2,
        "min_samples_l2": args.min_samples_l2,
        "counts": {"papers": len(papers)},
        "dimensions": {},
    }

    def build_meta_for_dimension(
        dim: str,
        texts: list[str],
        emb_normed: np.ndarray,
        l1_labels: list[int],
        l2_labels: list[int],
        is_keywords: bool,
    ) -> dict[str, Any]:
        # group members
        l1_groups: dict[int, list[int]] = defaultdict(list)
        for i, lab in enumerate(l1_labels):
            l1_groups[lab].append(i)

        dim_meta: dict[str, Any] = {"l1": {}, "l2": {}}

        for l1_id, members in l1_groups.items():
            if l1_id == -1:
                label = "Other"
            elif is_keywords:
                tops = top_keywords_for_cluster(members, paper_keywords, topn=4)
                label = " / ".join(tops) if tops else "Topic"
            else:
                label = pick_representative_text(members, texts, emb_normed)

            dim_meta["l1"][str(l1_id)] = {
                "id": int(l1_id),
                "label": label,
                "size": len(members),
                "example_paper_ids": [papers[i]["id"] for i in members[:5]],
            }

            # L2 within this L1
            l2_groups: dict[int, list[int]] = defaultdict(list)
            for i in members:
                l2_groups[l2_labels[i]].append(i)

            for l2_id, l2_members in l2_groups.items():
                if l2_id == -1:
                    l2_label = "Other"
                elif is_keywords:
                    tops = top_keywords_for_cluster(l2_members, paper_keywords, topn=4)
                    l2_label = " / ".join(tops) if tops else label
                else:
                    l2_label = pick_representative_text(l2_members, texts, emb_normed)

                dim_meta["l2"].setdefault(str(l1_id), {})[str(l2_id)] = {
                    "id": int(l2_id),
                    "label": l2_label,
                    "size": len(l2_members),
                    "example_paper_ids": [papers[i]["id"] for i in l2_members[:5]],
                }

        return dim_meta

    cluster_meta["dimensions"]["keywords"] = build_meta_for_dimension(
        "keywords", kw_texts, kw_emb, kw_l1, kw_l2, is_keywords=True
    )
    cluster_meta["dimensions"]["method"] = build_meta_for_dimension(
        "method", method_texts, method_emb, m_l1, m_l2, is_keywords=False
    )
    cluster_meta["dimensions"]["result"] = build_meta_for_dimension(
        "result", result_texts, result_emb, r_l1, r_l2, is_keywords=False
    )
    cluster_meta["dimensions"]["contribution"] = build_meta_for_dimension(
        "contribution", contrib_texts, contrib_emb, c_l1, c_l2, is_keywords=False
    )

    # 5) per-paper labels
    labels: dict[str, Any] = {}
    for i, p in enumerate(papers):
        pid = p["id"]
        base: dict[str, Any] = {}
        old = existing_labels.get(pid)
        if isinstance(old, dict):
            base.update(old)

        base.update(
            {
            "keywords": {"l1": int(kw_l1[i]), "l2": int(kw_l2[i])},
            "method": {"l1": int(m_l1[i]), "l2": int(m_l2[i])},
            "result": {"l1": int(r_l1[i]), "l2": int(r_l2[i])},
            "contribution": {"l1": int(c_l1[i]), "l2": int(c_l2[i])},
            }
        )

        labels[pid] = base

    # 6) write outputs
    (out_dir / "papers.json").write_text(
        json.dumps(papers, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "cluster_meta.json").write_text(
        json.dumps(cluster_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    app_cfg = {
        "default": {
            "columns": ["stage", "keywords", "method"],
            "levels": ["l1", "l1", "l1"],
        },
        "options": {
            "dimensions": ["stage", "keywords", "method", "result", "contribution"],
            "levels": ["l1", "l2"],
        },
    }
    (out_dir / "app_config.json").write_text(
        json.dumps(app_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"papers={len(papers)}")
    print(f"wrote: {out_dir / 'papers.json'}")
    print(f"wrote: {out_dir / 'labels.json'}")
    print(f"wrote: {out_dir / 'cluster_meta.json'}")
    print(f"wrote: {out_dir / 'app_config.json'}")


if __name__ == "__main__":
    main()
