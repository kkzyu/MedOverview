import json
from pathlib import Path

src = Path(r"F:\202507\AIH_SURVEY\medical_2024_2025.jsonl")
out_keywords = Path(r"F:\202507\AIH_SURVEY\medical_2024_2025_keywords.jsonl")
out_clusters = Path(r"F:\202507\AIH_SURVEY\medical_2024_2025_keyword_clusters.json")

records = []
keywords_list = []

with src.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        kws = obj.get("keywords")
        if isinstance(kws, list) and kws:
            keywords_list.append(kws)
            records.append({
                "venue": obj.get("venue"),
                "year": obj.get("year"),
                "full_title": obj.get("full_title"),
                "keywords": kws,
            })

with out_keywords.open("w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def norm_kw(k: str) -> str:
    if not isinstance(k, str):
        return ""
    return " ".join(k.strip().lower().split())


norm_sets = [set(filter(None, (norm_kw(k) for k in kws))) for kws in keywords_list]

n = len(norm_sets)
parent = list(range(n))


def find(x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(a: int, b: int) -> None:
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union_sz = len(a | b)
    return inter / union_sz if union_sz else 0.0


threshold = 0.5
for i in range(n):
    for j in range(i + 1, n):
        if jaccard(norm_sets[i], norm_sets[j]) >= threshold:
            union(i, j)

clusters = {}
for i in range(n):
    r = find(i)
    clusters.setdefault(r, []).append(i)

cluster_list = []
for idxs in clusters.values():
    freq: dict[str, int] = {}
    for i in idxs:
        for k in norm_sets[i]:
            freq[k] = freq.get(k, 0) + 1
    rep = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    rep_keywords = [k for k, _ in rep[:10]]
    members = []
    for i in idxs:
        members.append({
            "keywords": keywords_list[i],
            "normalized": sorted(norm_sets[i]),
        })
    cluster_list.append({
        "size": len(idxs),
        "representative_keywords": rep_keywords,
        "members": members,
    })

cluster_list.sort(key=lambda c: (-c["size"], c["representative_keywords"]))

out = {
    "source": str(src),
    "threshold": threshold,
    "total_records_with_keywords": n,
    "clusters": cluster_list,
}

out_clusters.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

print(
    f"wrote {out_keywords} and {out_clusters}; records={n}, clusters={len(cluster_list)}"
)
