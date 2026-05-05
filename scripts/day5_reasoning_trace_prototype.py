#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import uuid
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
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_claim(claim_id: str, text: str, direction: str, impact_bps: int, evidence_ref_ids: list[str], unsupported: bool = False, unsupported_reason: str | None = None) -> dict[str, Any]:
    return {
        'claim_id': claim_id,
        'text': text,
        'direction': direction,
        'impact_bps': impact_bps,
        'evidence_ref_ids': evidence_ref_ids,
        'unsupported': unsupported,
        'unsupported_reason': unsupported_reason,
    }


def derive_audit_flags(audit_row: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    raw = audit_row.get('audit_flags', [])
    if isinstance(raw, list):
        flags.extend(raw)

    note = str(audit_row.get('missing_source_tiers_or_notes', '')).lower()
    if 'sparse_slow_backbone' in note:
        flags.append('sparse_slow_backbone')
    if 'source_tier_mismatch' in note:
        flags.append('source_tier_mismatch')
    if str(audit_row.get('fast_signal_thin', '')).lower() == 'true':
        flags.append('fast_signal_thin')
    if str(audit_row.get('slow_backbone_thin', '')).lower() == 'true':
        flags.append('slow_backbone_thin')
    if str(audit_row.get('coverage_gate_bucket_status', '')).lower() == 'provisionally_clear_with_flags':
        flags.append('coverage_bucket_with_flags')
    return sorted(set(flags))


def refs_by_tier(rep: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for r in rep.get('evidence_refs', []):
        t = str(r.get('tier', 'unknown'))
        out.setdefault(t, []).append(str(r.get('ref_id')))
    return out


def domain_priority(domain: str) -> tuple[list[str], list[str]]:
    if domain == 'economics':
        return (['official_macro'], ['wikipedia_wikidata', 'gdelt_global_news'])
    if domain == 'culture':
        return (['awards_boxoffice_chart', 'wikipedia_wikidata'], ['gdelt_global_news'])
    return (['wikipedia_wikidata', 'gdelt_global_news'], ['official_politics', 'market_metadata'])


def select_refs_with_priority(rep: dict[str, Any], domain: str, need_primary: int = 3) -> dict[str, Any]:
    by_tier = refs_by_tier(rep)
    primary_tiers, secondary_tiers = domain_priority(domain)

    primary_refs: list[str] = []
    for t in primary_tiers:
        primary_refs.extend(by_tier.get(t, []))

    secondary_refs: list[str] = []
    for t in secondary_tiers:
        secondary_refs.extend(by_tier.get(t, []))

    fallback_used = len(primary_refs) < need_primary and len(secondary_refs) > 0
    selected = (primary_refs[:need_primary] + secondary_refs[: max(0, need_primary - len(primary_refs))])[:need_primary]

    return {
        'selected': selected,
        'primary_refs': primary_refs,
        'secondary_refs': secondary_refs,
        'primary_tiers': primary_tiers,
        'secondary_tiers': secondary_tiers,
        'fallback_used': fallback_used,
        'fallback_reason': 'insufficient_primary_tier_coverage' if fallback_used else None,
    }


def build_trace(rep: dict[str, Any], audit_row: dict[str, Any]) -> dict[str, Any]:
    qid = rep['query_id']
    domain = str(audit_row.get('domain'))
    coverage_flags = list(rep.get('coverage_flags', []))
    audit_flags = derive_audit_flags(audit_row)
    merged_flags = sorted(set(coverage_flags + audit_flags))
    sparse = any(f in SPARSE_FLAGS for f in merged_flags)

    selection = select_refs_with_priority(rep, domain, need_primary=3)
    primary = selection['primary_refs']
    secondary = selection['secondary_refs']

    claims: list[dict[str, Any]] = []

    if domain == 'economics':
        claims.append(build_claim('c1', 'Official macro releases provide direct signal for policy/indicator threshold mechanics.', 'up', 900, primary[:2] or selection['selected'][:2]))
        claims.append(build_claim('c2', 'Institutional macro linkage supports mechanism plausibility.', 'up', 250, primary[2:3] or selection['selected'][:1]))
        claims.append(build_claim('c3', 'Residual cross-series uncertainty tempers confidence.', 'down', 250, secondary[:1] or primary[:1]))
    elif domain == 'culture':
        qtext = str(rep.get('question_text', '')).lower()
        if 'gross' in qtext or 'billboard' in qtext or '#1' in qtext:
            c1_txt = 'Audience-performance trajectory from structured chart/box-office records is the primary driver.'
        else:
            c1_txt = 'Institutional award record (nominee/winner scaffold) is the primary driver.'

        claims.append(build_claim('c1', c1_txt, 'up', 700, primary[:2] or selection['selected'][:2]))
        claims.append(build_claim('c2', 'Competitor-field uncertainty remains and can offset single-signal dominance.', 'down', 350, primary[2:3] or secondary[:1] or selection['selected'][:1]))
        claims.append(build_claim('c3', 'Any external buzz/polling narrative not anchored in culture-tier refs is unsupported.', 'neutral', 0, [], unsupported=True, unsupported_reason='External narrative not present in evidence_refs'))
    else:
        claims.append(build_claim('c1', 'Backbone entities establish principal institutional actors.', 'up', 350, primary[:2] or selection['selected'][:2]))
        claims.append(build_claim('c2', 'News-context uncertainty can reduce confidence when signal is thin/noisy.', 'down', 600, primary[2:3] or secondary[:1] or selection['selected'][:1]))
        claims.append(build_claim('c3', 'External polling/model claims are unsupported unless present in evidence_refs.', 'neutral', 0, [], unsupported=True, unsupported_reason='External polling/model inputs not present in evidence_refs'))

    if sparse:
        decision = {
            'mode': 'abstain',
            'probability_yes': None,
            'confidence': 'low',
            'abstention_reason': 'Sparse-coverage flags indicate thin source support for this query at cutoff.',
        }
    else:
        net_bps = sum(c['impact_bps'] if c['direction'] == 'up' else -c['impact_bps'] if c['direction'] == 'down' else 0 for c in claims if not c['unsupported'])
        p_yes = max(0.05, min(0.95, 0.5 + net_bps / 10000.0))
        decision = {
            'mode': 'probability',
            'probability_yes': round(p_yes, 3),
            'confidence': 'medium' if abs(net_bps) < 1200 else 'high',
            'abstention_reason': None,
        }

    return {
        'trace_id': f'trace_day5::{qid}::{uuid.uuid4().hex[:8]}',
        'created_at': now_z(),
        'query_id': qid,
        'domain': domain,
        'cutoff_t': rep.get('cutoff_t'),
        'question_text': rep.get('question_text'),
        'schema_version': 'day5_trace_v1',
        'inputs': {
            'representation_id': f'R_det::{qid}',
            'coverage_flags_representation': coverage_flags,
            'coverage_flags_audit': audit_flags,
            'coverage_flags_merged': merged_flags,
            'evidence_ref_count': len(rep.get('evidence_refs', [])),
            'evidence_selection': {
                'primary_tiers': selection['primary_tiers'],
                'secondary_tiers': selection['secondary_tiers'],
                'fallback_used': selection['fallback_used'],
                'fallback_reason': selection['fallback_reason'],
            },
        },
        'claims': claims,
        'decision': decision,
        'unsupported_claim_count': sum(1 for c in claims if c.get('unsupported')),
        'primary_tiers_used': selection['primary_tiers'],
        'fallback_used': selection['fallback_used'],
    }


def validate_trace(trace: dict[str, Any], rep: dict[str, Any]) -> dict[str, Any]:
    rep_ref_ids = {r.get('ref_id') for r in rep.get('evidence_refs', [])}
    required_top = {'trace_id', 'query_id', 'schema_version', 'claims', 'decision', 'inputs'}
    has_required = required_top.issubset(trace.keys())

    prob_mode = trace.get('decision', {}).get('mode') == 'probability'
    moving_claims = [c for c in trace.get('claims', []) if c.get('impact_bps', 0) != 0 and not c.get('unsupported')]

    all_moving_have_refs = all(bool(c.get('evidence_ref_ids')) for c in moving_claims)
    all_refs_valid = all(rid in rep_ref_ids for c in moving_claims for rid in c.get('evidence_ref_ids', []))

    unsupported_claims = [c for c in trace.get('claims', []) if c.get('unsupported')]
    unsupported_have_reason = all(bool(c.get('unsupported_reason')) for c in unsupported_claims)

    sparse_flagged = any(f in SPARSE_FLAGS for f in trace['inputs'].get('coverage_flags_merged', []))
    abstention_trigger_ok = (trace['decision']['mode'] == 'abstain') if sparse_flagged else True

    return {
        'query_id': trace['query_id'],
        'schema_match_required_fields': has_required,
        'probability_mode': prob_mode,
        'moving_claim_count': len(moving_claims),
        'all_probability_moving_claims_have_evidence_ref_id': all_moving_have_refs,
        'all_cited_evidence_ref_ids_valid': all_refs_valid,
        'unsupported_claim_count': len(unsupported_claims),
        'unsupported_claims_have_reason': unsupported_have_reason,
        'sparse_flagged': sparse_flagged,
        'abstention_trigger_ok': abstention_trigger_ok,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--representations', default='data/representations/seed30/day4_deterministic/seed30_deterministic_representations.jsonl')
    ap.add_argument('--audit', default='.context/polymarket_30_seed_coverage_audit.json')
    ap.add_argument('--query-ids', nargs='*', default=None)
    ap.add_argument('--all-queries', action='store_true')
    ap.add_argument('--out-dir', default='data/representations/seed30/day5_trace_prototype')
    args = ap.parse_args()

    root = Path.cwd()
    reps = {r['query_id']: r for r in load_jsonl(root / args.representations)}
    audit = {r['query_id']: r for r in json.loads((root / args.audit).read_text(encoding='utf-8'))}

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    traces = []
    validations = []
    if args.all_queries:
        query_ids = sorted(reps.keys())
    else:
        query_ids = args.query_ids or []
    if not query_ids:
        raise SystemExit('Provide --all-queries or --query-ids ...')

    for qid in query_ids:
        rep = reps[qid]
        a = audit[qid]
        t = build_trace(rep, a)
        v = validate_trace(t, rep)
        traces.append(t)
        validations.append(v)

    full = len(query_ids) >= 30
    trace_name = 'seed30_full_traces.jsonl' if full else 'six_query_traces.jsonl'
    val_name = 'seed30_full_trace_validation.json' if full else 'six_query_trace_validation.json'
    trace_path = out_dir / trace_name
    val_path = out_dir / val_name

    with trace_path.open('w', encoding='utf-8') as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + '\n')

    val_obj = {'created_at': now_z(), 'query_ids': args.query_ids, 'validations': validations}
    val_path.write_text(json.dumps(val_obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    print(json.dumps({'trace_path': str(trace_path.resolve()), 'validation_path': str(val_path.resolve()), 'validations': validations}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
