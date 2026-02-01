import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def record_key(obj: Dict[str, Any]) -> str:
    # Prefer stable identifiers across sources.
    for k in ["openreview_forum_id", "openreview_id", "url"]:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return f"{k}:{v.strip()}"
    title = obj.get("full_title") or obj.get("title") or ""
    year = obj.get("year")
    return f"title:{str(title).strip()}|year:{year}"


def main() -> None:
    p = argparse.ArgumentParser(description="Merge multiple JSONL sources with simple dedup.")
    p.add_argument("--inputs", nargs="+", required=True, help="Input JSONL files")
    p.add_argument("--out", required=True, help="Output merged JSONL")
    p.add_argument("--prefer", choices=["first", "last"], default="first", help="Which record wins on duplicates")
    args = p.parse_args()

    inputs = [Path(x) for x in args.inputs]
    out = Path(args.out)

    seen: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    for ip in inputs:
        if not ip.exists():
            raise SystemExit(f"input not found: {ip}")
        for obj in iter_jsonl(ip):
            if not isinstance(obj, dict):
                continue
            k = record_key(obj)
            if k not in seen:
                seen[k] = obj
                order.append(k)
            else:
                if args.prefer == "last":
                    seen[k] = obj

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for k in order:
            f.write(json.dumps(seen[k], ensure_ascii=False) + "\n")

    print(f"merged={len(order)} wrote={out}")


if __name__ == "__main__":
    main()
