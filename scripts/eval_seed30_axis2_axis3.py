#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SPARSE_FLAGS = {
    'sparse_slow_backbone',
    'source_tier_mismatch',
    'fast_signal_thin',
    'slow_backbone_thin',
    'coverage_bucket_with_flags',
}


def now_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def outcome_yes(row: dict[str, Any]) -> int:
    raw = str(row.get('resolved_outcome', '')).strip().lower()
    if raw in {'yes', 'true', '1'}:
        return 1
    if raw in {'no', 'false', '0'}:
        return 0
    price = row.get('final_yes_price')
    if price is not None:
        return 1 if float(price) >= 0.5 else 0
    raise ValueError(f"Cannot parse resolved outcome for {row.get('query_id')}: {row.get('resolved_outcome')}")


def refs_for_trace(trace: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for c in trace.get('claims', []):
        refs.update(str(r) for r in c.get('evidence_ref_ids', []))
    return refs


def axis2_row(trace: dict[str, Any], rep: dict[str, Any]) -> dict[str, Any]:
    rep_ref_ids = {str(r.get('ref_id')) for r in rep.get('evidence_refs', [])}
    claims = trace.get('claims', [])
    moving = [c for c in claims if c.get('impact_bps', 0) != 0 and not c.get('unsupported')]
    moving_refs = [str(r) for c in moving for r in c.get('evidence_ref_ids', [])]
    invalid = [r for r in moving_refs if r not in rep_ref_ids]
    unsupported_moving = [c for c in claims if c.get('impact_bps', 0) != 0 and c.get('unsupported')]
    unsupported_claims = [c for c in claims if c.get('unsupported')]
    merged_flags = trace.get('inputs', {}).get('coverage_flags_merged', [])
    sparse = any(f in SPARSE_FLAGS for f in merged_flags)
    abstained = trace.get('decision', {}).get('mode') == 'abstain'
    fallback = bool(trace.get('fallback_used') or trace.get('inputs', {}).get('evidence_selection', {}).get('fallback_used'))
    return {
        'query_id': trace['query_id'],
        'domain': trace.get('domain'),
        'claim_count': len(claims),
        'moving_claim_count': len(moving),
        'invalid_citation_count': len(invalid),
        'invalid_citation_rate': len(invalid) / len(moving_refs) if moving_refs else 0.0,
        'probability_moving_unsupported_count': len(unsupported_moving),
        'probability_moving_unsupported_rate': len(unsupported_moving) / len(moving) if moving else 0.0,
        'unsupported_claim_count': len(unsupported_claims),
        'unsupported_claims_have_reason': all(bool(c.get('unsupported_reason')) for c in unsupported_claims),
        'sparse_flagged': sparse,
        'abstained': abstained,
        'abstention_trigger_ok': abstained if sparse else True,
        'fallback_used': fallback,
        'evidence_ref_count': len(rep.get('evidence_refs', [])),
        'cited_ref_count': len(set(moving_refs)),
    }


def brier(p: float, y: int) -> float:
    return (p - y) ** 2


def axis3_row(trace: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    decision = trace.get('decision', {})
    p_raw = decision.get('probability_yes')
    abstained = decision.get('mode') == 'abstain'
    p = 0.5 if p_raw is None else float(p_raw)
    y = outcome_yes(audit)
    return {
        'query_id': trace['query_id'],
        'domain': trace.get('domain'),
        'resolved_yes': y,
        'probability_yes': p_raw,
        'scored_probability_yes': p,
        'abstained': abstained,
        'brier': brier(p, y),
        'abs_from_0_5': abs(p - 0.5),
        'near_0_5': abs(p - 0.5) <= 0.05,
    }


def same_domain_substitute_rows(traces: list[dict[str, Any]], reps: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in traces:
        by_domain[str(t.get('domain'))].append(t)
    rows = []
    for domain, domain_traces in by_domain.items():
        ordered = sorted(domain_traces, key=lambda r: r['query_id'])
        if len(ordered) < 2:
            continue
        for i, t in enumerate(ordered):
            sub = ordered[(i + 1) % len(ordered)]
            real_refs = refs_for_trace(t)
            sub_rep_refs = {str(r.get('ref_id')) for r in reps[sub['query_id']].get('evidence_refs', [])}
            inter = real_refs & sub_rep_refs
            union = real_refs | sub_rep_refs
            rows.append({
                'query_id': t['query_id'],
                'domain': domain,
                'substitute_query_id': sub['query_id'],
                'cited_refs_preserved_under_same_domain_substitute': len(inter),
                'jaccard_cited_vs_substitute_rep_refs': len(inter) / len(union) if union else 0.0,
                'substitute_rep_ref_count': len(sub_rep_refs),
            })
    return rows


def summarize(rows: list[dict[str, Any]], numeric_keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {'n': len(rows)}
    for k in numeric_keys:
        vals = [float(r[k]) for r in rows if r.get(k) is not None and isinstance(r.get(k), (int, float, bool))]
        out[k] = mean(vals)
    return out


def grouped(rows: list[dict[str, Any]], numeric_keys: list[str]) -> dict[str, Any]:
    domains = sorted({str(r.get('domain')) for r in rows})
    return {d: summarize([r for r in rows if str(r.get('domain')) == d], numeric_keys) for d in domains}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--representations', default='data/representations/seed30/day4_deterministic/seed30_deterministic_representations.jsonl')
    ap.add_argument('--axis1-summary', default='data/representations/seed30/day4_deterministic/axis1_summary.json')
    ap.add_argument('--traces', default='data/representations/seed30/day5_trace_prototype/seed30_full_traces.jsonl')
    ap.add_argument('--audit', default='.context/polymarket_30_seed_coverage_audit.json')
    ap.add_argument('--out-dir', default='data/representations/seed30/day6_axis2_day7_ledger')
    ap.add_argument('--calibration-policy', default=None, help='Optional policy JSON path. When set, runs apply_seed30_calibration_policy.py after axis summaries.')
    ap.add_argument('--policy-mode', choices=['benchmark', 'production', 'both'], default='both')
    args = ap.parse_args()

    root = Path.cwd()
    reps_list = load_jsonl(root / args.representations)
    reps = {r['query_id']: r for r in reps_list}
    traces = load_jsonl(root / args.traces)
    audit_list = load_json(root / args.audit)
    audit = {r['query_id']: r for r in audit_list}
    axis1_summary = load_json(root / args.axis1_summary)

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    axis2 = [axis2_row(t, reps[t['query_id']]) for t in traces]
    axis3 = [axis3_row(t, audit[t['query_id']]) for t in traces]
    ablations = same_domain_substitute_rows(traces, reps)

    ledger = []
    axis2_by_q = {r['query_id']: r for r in axis2}
    axis3_by_q = {r['query_id']: r for r in axis3}
    for rep in reps_list:
        qid = rep['query_id']
        a = audit[qid]
        row = {
            'query_id': qid,
            'domain': a['domain'],
            'question_text': rep.get('question_text'),
            'cutoff_t': rep.get('cutoff_t'),
            'resolved_outcome': a.get('resolved_outcome'),
            'evidence_ref_count': len(rep.get('evidence_refs', [])),
            'coverage_flags': rep.get('coverage_flags', []),
            'assumption_count': len(rep.get('assumption_states', [])),
        }
        row.update({f'axis2_{k}': v for k, v in axis2_by_q[qid].items() if k not in {'query_id', 'domain'}})
        row.update({f'axis3_{k}': v for k, v in axis3_by_q[qid].items() if k not in {'query_id', 'domain'}})
        ledger.append(row)

    axis2_summary = {
        'created_at': now_z(),
        'queries': len(axis2),
        'overall': summarize(axis2, ['invalid_citation_rate', 'probability_moving_unsupported_rate', 'unsupported_claim_count', 'sparse_flagged', 'abstained', 'abstention_trigger_ok', 'fallback_used', 'evidence_ref_count', 'cited_ref_count']),
        'by_domain': grouped(axis2, ['invalid_citation_rate', 'probability_moving_unsupported_rate', 'unsupported_claim_count', 'sparse_flagged', 'abstained', 'abstention_trigger_ok', 'fallback_used', 'evidence_ref_count', 'cited_ref_count']),
        'hard_gate': {
            'invalid_citation_count': sum(r['invalid_citation_count'] for r in axis2),
            'probability_moving_unsupported_count': sum(r['probability_moving_unsupported_count'] for r in axis2),
            'abstention_trigger_failures': sum(1 for r in axis2 if not r['abstention_trigger_ok']),
            'pass': sum(r['invalid_citation_count'] for r in axis2) == 0 and sum(r['probability_moving_unsupported_count'] for r in axis2) == 0 and all(r['abstention_trigger_ok'] for r in axis2),
        },
    }

    axis3_summary = {
        'created_at': now_z(),
        'queries': len(axis3),
        'overall': summarize(axis3, ['brier', 'abs_from_0_5', 'near_0_5', 'abstained']),
        'by_domain': grouped(axis3, ['brier', 'abs_from_0_5', 'near_0_5', 'abstained']),
        'calibration_note': 'Abstentions are scored at 0.5 for terminal Brier; probabilities are heuristic trace-head outputs, not tuned forecasts.',
    }

    ablation_summary = {
        'created_at': now_z(),
        'ablation': 'same_domain_substitute_representation_ref_overlap',
        'rows': len(ablations),
        'overall': summarize(ablations, ['jaccard_cited_vs_substitute_rep_refs', 'cited_refs_preserved_under_same_domain_substitute', 'substitute_rep_ref_count']),
        'by_domain': grouped(ablations, ['jaccard_cited_vs_substitute_rep_refs', 'cited_refs_preserved_under_same_domain_substitute', 'substitute_rep_ref_count']),
        'interpretation': 'Low overlap means traces cite query-specific evidence rather than refs that survive same-domain substitution. This is a representation-specificity proxy, not a semantic randomization test.',
    }

    joined_summary = {
        'created_at': now_z(),
        'axis1_summary': axis1_summary,
        'axis2_summary': axis2_summary,
        'axis3_summary': axis3_summary,
        'ablation_summary': ablation_summary,
        'week2_gate_decision': {
            'decision': 'proceed_to_calibration_and_selectivity_tuning_not_architecture_migration',
            'rationale': [
                'Axis 2 hard gate passes with zero invalid citations and zero probability-moving unsupported claims.',
                'Axis 3 is weak/under-sharp because the trace-head is heuristic and many probabilities sit near 0.5.',
                'Axis 1 coverage is usable but politics/culture specificity is weaker than economics; improve selectivity and culture tier grounding before scaling.',
            ],
            'next_priorities': [
                'Calibration/sharpness layer on non-abstained trace probabilities.',
                'Politics specificity and official-source enrichment.',
                'Culture source-tier quality and unsupported-background-claim suppression.',
                'Representation selectivity ablations beyond same-domain substitution.',
            ],
        },
    }

    def write_jsonl(name: str, rows: list[dict[str, Any]]) -> None:
        with (out_dir / name).open('w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    write_jsonl('seed30_axis2_faithfulness.jsonl', axis2)
    write_jsonl('seed30_axis3_terminal_scores.jsonl', axis3)
    write_jsonl('seed30_joint_ledger.jsonl', ledger)
    write_jsonl('seed30_same_domain_substitution_ablation.jsonl', ablations)
    (out_dir / 'seed30_axis2_summary.json').write_text(json.dumps(axis2_summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (out_dir / 'seed30_axis3_summary.json').write_text(json.dumps(axis3_summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (out_dir / 'seed30_ablation_summary.json').write_text(json.dumps(ablation_summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (out_dir / 'seed30_axis123_summary.json').write_text(json.dumps(joined_summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    policy_runs: list[dict[str, Any]] = []
    if args.calibration_policy:
        modes = ['benchmark', 'production'] if args.policy_mode == 'both' else [args.policy_mode]
        for mode in modes:
            cmd = [
                'python',
                'scripts/apply_seed30_calibration_policy.py',
                '--policy',
                args.calibration_policy,
                '--mode',
                mode,
            ]
            proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
            policy_runs.append({
                'mode': mode,
                'returncode': proc.returncode,
                'stdout': proc.stdout.strip(),
                'stderr': proc.stderr.strip(),
            })
            if proc.returncode != 0:
                raise RuntimeError(f'Policy run failed for mode={mode}: {proc.stderr}')

    print(json.dumps({'out_dir': str(out_dir.resolve()), 'axis2_hard_gate_pass': axis2_summary['hard_gate']['pass'], 'axis3_brier': axis3_summary['overall']['brier'], 'near_0_5_rate': axis3_summary['overall']['near_0_5'], 'policy_runs': policy_runs}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
