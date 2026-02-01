import argparse
import json
import os
from pathlib import Path
from typing import Any

from deepseek_client import DeepSeekClient


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def build_stage_meta(taxonomy: dict[str, Any], labels_by_paper: dict[str, Any]) -> dict[str, Any]:
    l1 = taxonomy["l1"]

    # ECharts app expects a meta-like structure (l1 map, l2 nested maps)
    meta = {"l1": {}, "l2": {}}

    # count sizes from labels
    counts_l1: dict[str, int] = {}
    counts_l2: dict[tuple[str, str], int] = {}

    for pid, lab in labels_by_paper.items():
        stages = lab.get("stage") or {}
        l1s = stages.get("l1") or []
        l2s = stages.get("l2") or []

        for x in l1s:
            counts_l1[x] = counts_l1.get(x, 0) + 1
        for y in l2s:
            if "::" in y:
                p, c = y.split("::", 1)
            else:
                p, c = "Other", y
            counts_l2[(p, c)] = counts_l2.get((p, c), 0) + 1

    for l1_name, info in l1.items():
        meta["l1"][l1_name] = {
            "id": l1_name,
            "label": l1_name,
            "desc": info.get("desc", ""),
            "size": int(counts_l1.get(l1_name, 0)),
        }

        meta["l2"].setdefault(l1_name, {})
        for l2_name in info.get("l2", []):
            meta["l2"][l1_name][l2_name] = {
                "id": l2_name,
                "label": l2_name,
                "size": int(counts_l2.get((l1_name, l2_name), 0)),
            }

    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--papers",
        default=r"F:\202507\AIH_SURVEY\data\papers.json",
        help="Path to papers.json",
    )
    parser.add_argument(
        "--labels",
        default=r"F:\202507\AIH_SURVEY\data\labels.json",
        help="Path to labels.json (will be updated with stage labels)",
    )
    parser.add_argument(
        "--taxonomy",
        default=r"F:\202507\AIH_SURVEY\data\stage_taxonomy.json",
        help="Stage taxonomy",
    )
    parser.add_argument(
        "--out-stage-meta",
        default=r"F:\202507\AIH_SURVEY\data\stage_meta.json",
        help="Output stage meta for UI",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only first N papers (0 means all)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=0,
        help="Only label papers of this year (0 means all)",
    )
    parser.add_argument(
        "--venue",
        default="",
        help="Only label papers whose venue contains this substring (case-insensitive). Empty means all.",
    )
    parser.add_argument(
        "--deepseek-api-key",
        default="",
        help="DeepSeek API Key (preferred: set env DEEPSEEK_API_KEY; avoid committing keys)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing stage labels if present",
    )
    args = parser.parse_args()

    papers_path = Path(args.papers)
    labels_path = Path(args.labels)
    taxonomy_path = Path(args.taxonomy)
    out_stage_meta = Path(args.out_stage_meta)

    papers = load_json(papers_path)
    labels = load_json(labels_path)
    taxonomy = load_json(taxonomy_path)

    # Allow passing key via CLI for convenience (still supports .env / env vars).
    if isinstance(args.deepseek_api_key, str) and args.deepseek_api_key.strip():
        os.environ["DEEPSEEK_API_KEY"] = args.deepseek_api_key.strip()

    client = DeepSeekClient.from_env()

    l1 = taxonomy["l1"]
    allowed_l1 = list(l1.keys())
    allowed_l2 = {
        k: list(v.get("l2", []))
        for k, v in l1.items()
    }

    system = (
        "You are a careful research assistant. "
        "You must output ONLY valid JSON with the requested schema. "
        "Use the provided taxonomy strictly. "
        "A paper can have multiple stages. Prefer 1-3 L1 stages. "
        "For L2, output as 'L1::L2'. "
        "If unsure, use 'Other'."
    )

    processed = 0
    venue_sub = str(args.venue or "").strip().lower()
    year_only = int(args.year or 0)

    def paper_selected(p: dict[str, Any]) -> bool:
        if year_only:
            try:
                if int(p.get("year") or 0) != year_only:
                    return False
            except Exception:
                return False
        if venue_sub:
            v = str(p.get("venue") or "").lower()
            if venue_sub not in v:
                return False
        return True

    for p in papers[: (args.limit or None)]:
        if not paper_selected(p):
            continue
        pid = p["id"]
        if (not args.overwrite) and labels.get(pid, {}).get("stage"):
            continue

        user = {
            "task": "Assign clinical workflow stage labels for this medical ML paper.",
            "taxonomy": {"allowed_l1": allowed_l1, "allowed_l2": allowed_l2},
            "paper": {
                "id": pid,
                "title": p.get("title"),
                "abstract": p.get("abstract"),
                "keywords": p.get("keywords"),
                "triple": p.get("triple"),
                "venue": p.get("venue"),
                "year": p.get("year"),
            },
            "output_schema": {
                "stage_l1": "list of L1 strings",
                "stage_l2": "list of strings in form 'L1::L2'",
                "confidence": "float 0..1",
                "rationale_cn": "one short Chinese sentence",
            },
        }

        resp = client.chat_json(system=system, user=json.dumps(user, ensure_ascii=False))

        stage_l1 = resp.get("stage_l1")
        stage_l2 = resp.get("stage_l2")
        if not isinstance(stage_l1, list):
            stage_l1 = ["Other"]
        stage_l1 = [x for x in stage_l1 if isinstance(x, str) and x in allowed_l1]
        if not stage_l1:
            stage_l1 = ["Other"]

        if not isinstance(stage_l2, list):
            stage_l2 = []
        stage_l2_clean: list[str] = []
        for x in stage_l2:
            if not isinstance(x, str) or "::" not in x:
                continue
            p1, c1 = x.split("::", 1)
            if p1 in allowed_l1 and c1 in allowed_l2.get(p1, []):
                stage_l2_clean.append(f"{p1}::{c1}")
        stage_l2_clean = list(dict.fromkeys(stage_l2_clean))

        # If we already have meaningful labels, drop the catch-all Other.
        if len(stage_l1) > 1 and "Other" in stage_l1:
            stage_l1 = [x for x in stage_l1 if x != "Other"]
        if len(stage_l2_clean) > 1 and "Other::Other" in stage_l2_clean:
            stage_l2_clean = [x for x in stage_l2_clean if x != "Other::Other"]

        # Keep L1 compact (prefer 1-3) to avoid too many multi-path edges in Sankey.
        stage_l1 = stage_l1[:3]

        confidence = resp.get("confidence")
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        rationale_cn = resp.get("rationale_cn")
        if not isinstance(rationale_cn, str):
            rationale_cn = ""

        labels.setdefault(pid, {})["stage"] = {
            "l1": stage_l1,
            "l2": stage_l2_clean,
            "confidence": confidence,
            "rationale_cn": rationale_cn,
        }

        processed += 1
        if processed % 10 == 0:
            save_json(labels_path, labels)
            print(f"checkpoint: updated {processed} papers")

    save_json(labels_path, labels)

    # build stage meta (counts + descriptions) for UI
    stage_meta = build_stage_meta(taxonomy, labels)
    out_stage_meta.write_text(
        json.dumps(stage_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"done. updated labels: {labels_path}")
    print(f"wrote stage meta: {out_stage_meta}")


if __name__ == "__main__":
    main()
