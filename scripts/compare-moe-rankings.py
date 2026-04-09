#!/usr/bin/env python3
"""Compare MoE ranking CSVs (full vs micro vs mmap).

Reads ranking files where each non-comment line is either:
  - expert_id
  - expert_id,total_mass,...

Outputs JSON with:
  - spearman rank correlation (against --truth)
  - recall@K
  - top-K overlap
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def load_ranking(path: Path) -> List[int]:
    ranking: List[int] = []
    seen = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("expert"):
            continue
        head = line.split(",", 1)[0].strip()
        try:
            expert = int(head)
        except ValueError as exc:
            raise ValueError(f"failed to parse expert id from {path}: {line!r}") from exc
        if expert in seen:
            raise ValueError(f"duplicate expert id {expert} in {path}")
        seen.add(expert)
        ranking.append(expert)
    if not ranking:
        raise ValueError(f"ranking file is empty or unreadable: {path}")
    return ranking


def validate_same_experts(rankings: Dict[str, List[int]]) -> None:
    items = list(rankings.items())
    base_name, base = items[0]
    base_set = set(base)
    for name, ranking in items[1:]:
        rset = set(ranking)
        if rset != base_set:
            missing = sorted(base_set - rset)[:8]
            extra = sorted(rset - base_set)[:8]
            raise ValueError(
                f"{name} does not match expert set from {base_name}; "
                f"missing={missing} extra={extra}"
            )


def spearman(a: Sequence[int], b: Sequence[int]) -> float:
    n = min(len(a), len(b))
    if n < 2:
        return 1.0
    rank_a = {expert: i for i, expert in enumerate(a)}
    sum_d2 = 0.0
    used = 0
    for i, expert in enumerate(b[:n]):
        if expert not in rank_a:
            continue
        d = rank_a[expert] - i
        sum_d2 += d * d
        used += 1
    if used < 2:
        return 0.0
    m = float(used)
    return 1.0 - (6.0 * sum_d2) / (m * (m * m - 1.0))


def recall_at_k(candidate: Sequence[int], truth: Sequence[int], k: int) -> float:
    if k <= 0:
        return 0.0
    k = min(k, len(candidate), len(truth))
    if k == 0:
        return 0.0
    a = set(candidate[:k])
    b = set(truth[:k])
    return len(a & b) / float(k)


@dataclass
class CompareInput:
    name: str
    path: Path


def parse_compare_items(items: Iterable[str]) -> List[CompareInput]:
    out: List[CompareInput] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--compare must use name=path format, got: {item!r}")
        name, p = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"blank compare name in: {item!r}")
        path = Path(p.strip()).expanduser()
        out.append(CompareInput(name=name, path=path))
    if not out:
        raise ValueError("at least one --compare is required")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare MoE ranking CSV files.")
    parser.add_argument(
        "--truth",
        required=True,
        type=Path,
        help="Ground-truth ranking CSV path (usually full analyze).",
    )
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        help="Ranking to compare in name=path form (repeatable).",
    )
    parser.add_argument(
        "--k",
        action="append",
        type=int,
        default=[],
        help="Recall@K values to compute (repeatable).",
    )
    parser.add_argument(
        "--min-experts",
        type=int,
        default=None,
        help="Optional K convenience value (added to --k list).",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=16,
        help="How many top experts to include in previews.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON file path (prints to stdout if omitted).",
    )
    args = parser.parse_args()

    truth_path = args.truth.expanduser()
    compare_items = parse_compare_items(args.compare)
    k_values = [k for k in args.k if k > 0]
    if args.min_experts and args.min_experts > 0:
        k_values.append(args.min_experts)
    if not k_values:
        k_values = [16, 32]
    k_values = sorted(set(k_values))

    rankings: Dict[str, List[int]] = {"truth": load_ranking(truth_path)}
    for item in compare_items:
        rankings[item.name] = load_ranking(item.path)
    validate_same_experts(rankings)

    truth = rankings["truth"]
    comparisons = []
    for item in compare_items:
        candidate = rankings[item.name]
        comparisons.append(
            {
                "name": item.name,
                "path": str(item.path),
                "ranking_len": len(candidate),
                "spearman_vs_truth": spearman(candidate, truth),
                "recall_at_k": {
                    str(k): recall_at_k(candidate, truth, k)
                    for k in k_values
                },
                "top_overlap": {
                    str(k): len(set(candidate[:k]) & set(truth[:k]))
                    for k in k_values
                },
                "preview_top": candidate[: args.preview],
            }
        )

    report = {
        "benchmark": "moe-ranking-compare",
        "truth": {
            "path": str(truth_path),
            "ranking_len": len(truth),
            "preview_top": truth[: args.preview],
        },
        "k_values": k_values,
        "comparisons": comparisons,
    }

    blob = json.dumps(report, indent=2)
    if args.output:
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(blob + "\n", encoding="utf-8")
        print(f"Wrote report to {output}")
    else:
        print(blob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

