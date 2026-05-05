#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def clip(p: float, eps: float = 1e-6) -> float:
    return min(1 - eps, max(eps, p))


def logit(p: float) -> float:
    p = clip(p)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def calibrate(p: float, spec: dict[str, Any]) -> float:
    kind = spec['type']
    if kind == 'temperature':
        T = float(spec['T'])
        return sigmoid(logit(p) / T)
    if kind == 'platt':
        a = float(spec['a'])
        b = float(spec['b'])
        return sigmoid(a * logit(p) + b)
    raise ValueError(f'Unknown calibrator type: {kind}')


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser(description='Apply benchmark/production calibration policy to seed30 axis3 outputs.')
    ap.add_argument('--policy', default='data/representations/seed30/day8_calibration_tuning/seed30_calibration_policy.json')
    ap.add_argument('--mode', choices=['benchmark', 'production'], default='benchmark')
    ap.add_argument('--out-dir', default='data/representations/seed30/day8_calibration_tuning')
    args = ap.parse_args()

    root = Path.cwd()
    policy = json.loads((root / args.policy).read_text(encoding='utf-8'))
    inp = policy['inputs']

    axis3 = load_jsonl(root / inp['axis3_scores_jsonl'])
    ledger = load_jsonl(root / inp['joint_ledger_jsonl'])
    qtext = {r['query_id']: r.get('question_text') for r in ledger}

    mode_cfg = policy['modes'][args.mode]
    calib_name = mode_cfg['calibrator']
    calib_spec = policy['models'][calib_name]
    tau = float(mode_cfg['tau'])
    abstain_score = float(mode_cfg.get('score_abstention_as', 0.5))

    out_rows = []
    brier_all = []
    brier_answered = []
    near05 = 0

    for r in axis3:
        qid = r['query_id']
        y = int(r['resolved_yes'])
        p0 = float(r['scored_probability_yes'])
        pc = calibrate(p0, calib_spec)
        abstain = abs(pc - 0.5) < tau
        p_scored = abstain_score if abstain else pc
        br = (p_scored - y) ** 2
        if not abstain:
            brier_answered.append((pc - y) ** 2)
        if abs(p_scored - 0.5) <= 0.05:
            near05 += 1
        brier_all.append(br)

        out_rows.append({
            'query_id': qid,
            'domain': r.get('domain'),
            'question_text': qtext.get(qid),
            'resolved_yes': y,
            'p_base': p0,
            'p_calibrated': pc,
            'mode': args.mode,
            'calibrator': calib_name,
            'tau': tau,
            'abstained': abstain,
            'p_scored': p_scored,
            'brier': br,
        })

    n = len(out_rows)
    answered_n = sum(1 for r in out_rows if not r['abstained'])
    summary = {
        'mode': args.mode,
        'calibrator': calib_name,
        'tau': tau,
        'n': n,
        'answered_n': answered_n,
        'coverage': answered_n / n if n else None,
        'abstain_rate': 1 - (answered_n / n if n else 0),
        'brier_with_abstain': mean(brier_all),
        'brier_answered_only': mean(brier_answered),
        'near_0_5_rate_scored': near05 / n if n else None,
        'policy_path': str((root / args.policy).resolve()),
    }

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f'seed30_axis3_policy_{args.mode}.jsonl'
    out_summary = out_dir / f'seed30_axis3_policy_{args.mode}_summary.json'

    with out_jsonl.open('w', encoding='utf-8') as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'rows': n, 'out_jsonl': str(out_jsonl.resolve()), 'out_summary': str(out_summary.resolve()), 'summary': summary}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
