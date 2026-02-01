import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if isinstance(obj, dict):
                yield obj


def norm_title(obj: Dict[str, Any]) -> str:
    title = obj.get("full_title") or obj.get("title") or obj.get("search_title") or ""
    title = str(title).strip().lower()
    title = re.sub(r"\s+", " ", title)
    # strip common punctuation to reduce trivial duplicates
    title = re.sub(r"[\-–—_:：,，.。;；!?！？()\[\]{}<>《》\"'“”‘’]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def score_record(obj: Dict[str, Any]) -> Tuple[int, int, int]:
    """Higher is better."""
    abstract = obj.get("abstract")
    kws = obj.get("keywords")
    tri = obj.get("triple")
    has_abs = 1 if isinstance(abstract, str) and abstract.strip() else 0
    has_kws = 1 if isinstance(kws, list) and any(isinstance(x, str) and x.strip() for x in kws) else 0
    has_tri = 1 if isinstance(tri, dict) and any(isinstance(v, str) and v.strip() for v in tri.values()) else 0
    # prefer longer abstracts (rough proxy for completeness)
    abs_len = len(abstract.strip()) if isinstance(abstract, str) else 0
    return (has_abs, has_kws + has_tri, abs_len)


def main() -> None:
    p = argparse.ArgumentParser(description="Deduplicate a JSONL file by (normalized) title.")
    p.add_argument("--in", dest="inp", required=True, help="Input JSONL")
    p.add_argument("--out", required=True, help="Output JSONL")
    p.add_argument(
        "--prefer",
        choices=["first", "best"],
        default="best",
        help="On duplicate titles, keep the first record or the 'best' record (default best).",
    )
    args = p.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    if not inp.exists():
        raise SystemExit(f"input not found: {inp}")

    kept: Dict[str, Dict[str, Any]] = {}
    order: list[str] = []

    total = 0
    dup = 0

    for obj in iter_jsonl(inp):
        total += 1
        k = norm_title(obj)
        if not k:
            # keep empty-title items as unique by line index
            k = f"__empty_title__:{total}"

        if k not in kept:
            kept[k] = obj
            order.append(k)
            continue

        dup += 1
        if args.prefer == "first":
            continue

        # prefer more complete record
        if score_record(obj) > score_record(kept[k]):
            kept[k] = obj

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for k in order:
            f.write(json.dumps(kept[k], ensure_ascii=False) + "\n")

    print(f"total={total} deduped={len(order)} removed={dup} wrote={out}")


if __name__ == "__main__":
    main()
