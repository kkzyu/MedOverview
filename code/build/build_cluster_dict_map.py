import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def norm_text(s: str) -> str:
    return " ".join(str(s or "").lower().split())


def score_category(text: str, keywords: list[str]) -> int:
    t = norm_text(text)
    score = 0
    for kw in keywords:
        k = norm_text(kw)
        if not k:
            continue
        if k in t:
            # longer keyword slightly higher weight
            score += 2 if len(k) >= 6 else 1
    return score


def pick_type(text: str, types: list[dict[str, Any]]) -> dict[str, Any]:
    best = None
    best_score = -1
    for tp in types:
        s = score_category(text, tp.get("keywords") or [])
        if s > best_score:
            best_score = s
            best = tp
    if best is None:
        return {"id": "other", "label_cn": "综合/其它", "description_cn": "无法可靠归类，或跨多类主题。"}
    # if no keyword matched at all, fall back
    if best_score <= 0:
        return {"id": "other", "label_cn": "综合/其它", "description_cn": "无法可靠归类，或跨多类主题。"}
    return best


def cluster_text(info: dict[str, Any]) -> str:
    parts = []
    if isinstance(info.get("llm"), dict):
        parts.append(info["llm"].get("name_cn") or "")
        parts.append(info["llm"].get("description_cn") or "")
        tags = info["llm"].get("tags")
        if isinstance(tags, list):
            parts.extend([str(x) for x in tags if isinstance(x, str)])
    parts.append(info.get("label") or "")
    return "\n".join([p for p in parts if p])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cluster-meta-llm",
        default=r"F:\202507\AIH_SURVEY\data\cluster_meta_llm.json",
    )
    parser.add_argument(
        "--dictionary",
        default=r"F:\202507\AIH_SURVEY\data\label_dictionary.json",
    )
    parser.add_argument(
        "--out",
        default=r"F:\202507\AIH_SURVEY\data\cluster_label_dict_map.json",
        help="Mapping file consumed by frontend (does NOT overwrite cluster_meta_llm.json)",
    )
    args = parser.parse_args()

    meta = load_json(Path(args.cluster_meta_llm))
    dic = load_json(Path(args.dictionary))

    result_types = dic.get("result_types") or []
    method_types = dic.get("method_types") or []

    out: dict[str, Any] = {
        "version": 1,
        "source": {
            "cluster_meta_llm": str(Path(args.cluster_meta_llm)),
            "dictionary": str(Path(args.dictionary)),
        },
        "note_cn": "字典映射仅用于展示层切换，不覆盖原始簇名。",
        "dimensions": {
            "method": {"l1": {}},
            "result": {"l1": {}},
        },
    }

    dims = meta.get("dimensions") or {}
    for dim, types in [("method", method_types), ("result", result_types)]:
        d = dims.get(dim) or {}
        l1 = d.get("l1") or {}
        for cid, info in l1.items():
            if not isinstance(info, dict):
                continue
            # skip noise id
            try:
                if int(info.get("id")) == -1:
                    continue
            except Exception:
                continue

            text = cluster_text(info)
            tp = pick_type(text, types)
            out["dimensions"][dim]["l1"][str(info.get("id"))] = {
                "type_id": tp.get("id"),
                "type_label_cn": tp.get("label_cn"),
                "type_description_cn": tp.get("description_cn"),
                "score_hint": "keyword_match",
            }

    save_json(Path(args.out), out)
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
