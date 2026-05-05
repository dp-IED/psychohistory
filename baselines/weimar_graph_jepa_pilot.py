"""Small PIT-clean Graph-JEPA-style pilot on curated Weimar Republic tri-domain data.

This is intentionally a fast diagnostic, not a production training path. It asks whether a
masked-domain graph representation objective can recover useful next-year structure across
three domains: economic, cultural, and socio-political.

Run:
    PYTHONPATH=. python -m baselines.weimar_graph_jepa_pilot --out-dir data/runs/weimar_graph_jepa_pilot
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

DOMAINS = ("economic", "cultural", "socio_political")
FEATURES = (
    # Economic: inflation/currency stress, unemployment/industrial stress, foreign-credit/fiscal fragility.
    ("inflation_currency_stress", "unemployment_industrial_stress", "credit_fiscal_fragility"),
    # Cultural: avant-garde/media intensity, nationalist backlash, polarization in public sphere.
    ("avant_garde_media_intensity", "nationalist_cultural_backlash", "public_sphere_polarization"),
    # Socio-political: street violence, coalition fragility/emergency rule, extremist electoral strength.
    ("street_violence", "coalition_fragility_emergency_rule", "extremist_electoral_strength"),
)
FEATURE_DIM = 3
LATENT_DIM = 6


@dataclass(frozen=True)
class YearState:
    year: int
    matrix: np.ndarray  # shape: (3 domains, 3 features), values in [0, 1]
    events: tuple[str, ...]
    regime_label: str


def _m(rows: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.array(list(rows), dtype=np.float64)
    if arr.shape != (len(DOMAINS), FEATURE_DIM):
        raise ValueError(f"bad matrix shape: {arr.shape}")
    return arr


def build_weimar_states() -> list[YearState]:
    """Curated yearly PIT states, using only information observable by the end of each year.

    Values are deliberately coarse ordinal intensities rather than pseudo-precise facts. The
    point is to test the representation objective's behavior on a historically plausible
    multi-domain graph, not to claim measurement finality.
    """
    rows = [
        YearState(1919, _m([[0.55, 0.65, 0.55], [0.55, 0.35, 0.55], [0.85, 0.90, 0.55]]), ("Versailles settlement", "Spartacist/Kapp-era violence begins", "Weimar constitution"), "birth_crisis"),
        YearState(1920, _m([[0.62, 0.58, 0.62], [0.58, 0.42, 0.58], [0.92, 0.85, 0.60]]), ("Kapp Putsch", "Ruhr Red Army suppression", "early coalition instability"), "birth_crisis"),
        YearState(1921, _m([[0.70, 0.50, 0.72], [0.62, 0.45, 0.60], [0.70, 0.72, 0.62]]), ("London reparations schedule", "political assassinations continue", "Bauhaus/modernist visibility"), "reparations_pressure"),
        YearState(1922, _m([[0.86, 0.48, 0.85], [0.66, 0.50, 0.66], [0.76, 0.76, 0.66]]), ("Rapallo controversy", "Rathenau assassination", "inflation accelerates"), "inflation_polarization"),
        YearState(1923, _m([[1.00, 0.72, 0.98], [0.70, 0.62, 0.80], [0.96, 0.98, 0.78]]), ("Ruhr occupation", "hyperinflation", "Beer Hall Putsch", "Saxony/Thuringia crisis"), "hyperinflation_putsch"),
        YearState(1924, _m([[0.38, 0.42, 0.45], [0.74, 0.55, 0.52], [0.42, 0.50, 0.35]]), ("Rentenmark stabilization", "Dawes Plan", "relative political cooling"), "stabilization"),
        YearState(1925, _m([[0.32, 0.40, 0.48], [0.78, 0.58, 0.54], [0.36, 0.48, 0.38]]), ("Hindenburg elected", "Locarno diplomacy", "Neue Sachlichkeit expands"), "stabilization"),
        YearState(1926, _m([[0.28, 0.38, 0.44], [0.86, 0.58, 0.52], [0.30, 0.42, 0.32]]), ("League of Nations entry", "Berlin modernist culture high visibility", "credit-supported recovery"), "golden_years"),
        YearState(1927, _m([[0.30, 0.42, 0.50], [0.90, 0.62, 0.55], [0.32, 0.44, 0.34]]), ("urban mass culture expansion", "welfare/fiscal tensions", "relative parliamentary normality"), "golden_years"),
        YearState(1928, _m([[0.34, 0.46, 0.54], [0.88, 0.66, 0.58], [0.34, 0.46, 0.36]]), ("Grand coalition period", "SPD electoral strength", "modernist backlash remains present"), "late_stability"),
        YearState(1929, _m([[0.62, 0.62, 0.75], [0.84, 0.72, 0.66], [0.52, 0.62, 0.48]]), ("Young Plan conflict", "Wall Street crash transmission", "paramilitary politics intensify"), "crisis_reentry"),
        YearState(1930, _m([[0.82, 0.88, 0.90], [0.78, 0.82, 0.82], [0.72, 0.88, 0.76]]), ("Bruning presidential cabinet", "austerity/deflation", "Nazi electoral breakthrough"), "depression_radicalization"),
        YearState(1931, _m([[0.90, 0.96, 0.98], [0.72, 0.88, 0.88], [0.82, 0.94, 0.86]]), ("banking crisis", "Harzburg Front", "street clashes escalate"), "depression_radicalization"),
        YearState(1932, _m([[0.92, 1.00, 0.98], [0.68, 0.94, 0.96], [0.96, 1.00, 0.98]]), ("mass unemployment", "Papen coup in Prussia", "July/November election crisis"), "collapse"),
        YearState(1933, _m([[0.78, 0.86, 0.88], [0.48, 1.00, 1.00], [1.00, 1.00, 1.00]]), ("Hitler appointed", "Reichstag Fire decree", "Enabling Act", "Gleichschaltung"), "collapse"),
    ]
    return rows


def make_masked_input(matrix: np.ndarray, masked_domain: int) -> np.ndarray:
    x = matrix.copy()
    x[masked_domain, :] = 0.0
    mask = np.zeros((len(DOMAINS), 1), dtype=np.float64)
    mask[masked_domain, 0] = 1.0
    return np.concatenate([x, mask], axis=1).reshape(-1)


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def ridge_fit(x: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    xb = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    # Use augmented least squares instead of normal equations. The pilot design matrix
    # is small and partly collinear by construction (three masked-domain views per
    # year), so this is numerically cleaner and avoids warnings from forming X.T @ X.
    penalty = np.sqrt(lam) * np.eye(xb.shape[1])
    penalty[-1, -1] = 0.0
    aug_x = np.concatenate([xb, penalty], axis=0)
    aug_y = np.concatenate([y, np.zeros((xb.shape[1], y.shape[1]))], axis=0)
    coef, *_ = np.linalg.lstsq(aug_x, aug_y, rcond=None)
    return coef


def ridge_predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    xb = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    return xb @ w


def train_graph_jepa(
    x: np.ndarray,
    target_latent: np.ndarray,
    *,
    seed: int = 7,
    epochs: int = 3500,
    lr: float = 0.035,
    weight_decay: float = 2e-4,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Train a small non-contrastive context encoder + predictor with stopped target latents.

    x -> tanh(x W_enc) -> z_ctx -> z_ctx W_pred -> stopped target latent.
    """
    rng = np.random.default_rng(seed)
    w_enc = rng.normal(0.0, 0.18, size=(x.shape[1], LATENT_DIM))
    w_pred = rng.normal(0.0, 0.18, size=(LATENT_DIM, LATENT_DIM))
    losses: list[float] = []
    n = x.shape[0]
    for epoch in range(epochs):
        h_raw = x @ w_enc
        h = np.tanh(h_raw)
        pred = h @ w_pred
        err = pred - target_latent
        loss = float(np.mean(err**2) + weight_decay * (np.mean(w_enc**2) + np.mean(w_pred**2)))
        if epoch % 250 == 0 or epoch == epochs - 1:
            losses.append(loss)
        g_pred = (2.0 / n) * h.T @ err + 2.0 * weight_decay * w_pred / w_pred.size
        g_h = (2.0 / n) * err @ w_pred.T
        g_raw = g_h * (1.0 - h * h)
        g_enc = x.T @ g_raw + 2.0 * weight_decay * w_enc / w_enc.size
        w_pred -= lr * g_pred
        w_enc -= lr * g_enc
    return w_enc, w_pred, losses


def build_examples(states: list[YearState]) -> list[dict]:
    examples = []
    for i in range(len(states) - 1):
        now = states[i]
        nxt = states[i + 1]
        for domain_idx, domain in enumerate(DOMAINS):
            examples.append(
                {
                    "as_of_year": now.year,
                    "target_year": nxt.year,
                    "domain_idx": domain_idx,
                    "domain": domain,
                    "x": make_masked_input(now.matrix, domain_idx),
                    "target": nxt.matrix[domain_idx].copy(),
                    "current_same_domain": now.matrix[domain_idx].copy(),
                    "regime_label": nxt.regime_label,
                    "events_as_of": now.events,
                    "target_events": nxt.events,
                }
            )
    return examples


def run_pilot(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    states = build_weimar_states()
    examples = build_examples(states)

    train = [e for e in examples if e["as_of_year"] <= 1929]
    eval_ = [e for e in examples if e["as_of_year"] >= 1930]

    # Stopped target encoder: fixed random projection of target domain vector into latent space.
    rng = np.random.default_rng(123)
    target_proj = rng.normal(0.0, 1.0, size=(FEATURE_DIM, LATENT_DIM))

    x_train = np.stack([e["x"] for e in train])
    y_train = np.stack([e["target"] for e in train])
    z_train = y_train @ target_proj
    z_train = z_train / (np.linalg.norm(z_train, axis=1, keepdims=True) + 1e-9)

    x_eval = np.stack([e["x"] for e in eval_])
    y_eval = np.stack([e["target"] for e in eval_])
    z_eval = y_eval @ target_proj
    z_eval = z_eval / (np.linalg.norm(z_eval, axis=1, keepdims=True) + 1e-9)

    w_enc, w_pred, losses = train_graph_jepa(x_train, z_train)
    z_pred_eval = np.tanh(x_eval @ w_enc) @ w_pred
    z_pred_eval = z_pred_eval / (np.linalg.norm(z_pred_eval, axis=1, keepdims=True) + 1e-9)

    ridge_w = ridge_fit(x_train, y_train)
    y_ridge_eval = np.clip(ridge_predict(x_eval, ridge_w), 0.0, 1.0)
    z_ridge_eval = y_ridge_eval @ target_proj
    z_ridge_eval = z_ridge_eval / (np.linalg.norm(z_ridge_eval, axis=1, keepdims=True) + 1e-9)

    y_persist_eval = np.stack([e["current_same_domain"] for e in eval_])
    z_persist_eval = y_persist_eval @ target_proj
    z_persist_eval = z_persist_eval / (np.linalg.norm(z_persist_eval, axis=1, keepdims=True) + 1e-9)

    # Retrieval bank is train target latents. This tests whether predicted latent finds a similar historical next-state.
    z_bank = z_train
    bank_labels = [e["regime_label"] for e in train]
    bank_years = [e["target_year"] for e in train]
    bank_domains = [e["domain"] for e in train]

    rows = []
    for i, e in enumerate(eval_):
        sims = z_bank @ z_pred_eval[i]
        nn = int(np.argmax(sims))
        rows.append(
            {
                "as_of_year": e["as_of_year"],
                "target_year": e["target_year"],
                "domain": e["domain"],
                "actual_regime": e["regime_label"],
                "jepa_cosine_to_actual": cosine(z_pred_eval[i], z_eval[i]),
                "ridge_cosine_to_actual": cosine(z_ridge_eval[i], z_eval[i]),
                "persistence_cosine_to_actual": cosine(z_persist_eval[i], z_eval[i]),
                "jepa_nn_year": bank_years[nn],
                "jepa_nn_domain": bank_domains[nn],
                "jepa_nn_regime": bank_labels[nn],
                "jepa_nn_similarity": float(sims[nn]),
                "jepa_regime_match": bank_labels[nn] == e["regime_label"],
                "actual_target_vector": y_eval[i].round(4).tolist(),
                "ridge_target_vector": y_ridge_eval[i].round(4).tolist(),
                "persistence_target_vector": y_persist_eval[i].round(4).tolist(),
            }
        )

    def grouped(rows_: list[dict], key: str) -> dict:
        out = {}
        for k in sorted({r[key] for r in rows_}):
            subset = [r for r in rows_ if r[key] == k]
            out[k] = {
                "n": len(subset),
                "jepa_cosine": float(np.mean([r["jepa_cosine_to_actual"] for r in subset])),
                "ridge_cosine": float(np.mean([r["ridge_cosine_to_actual"] for r in subset])),
                "persistence_cosine": float(np.mean([r["persistence_cosine_to_actual"] for r in subset])),
                "jepa_regime_match_rate": float(np.mean([r["jepa_regime_match"] for r in subset])),
            }
        return out

    summary = {
        "pilot": "weimar_graph_jepa_tri_domain",
        "train_years_as_of": [min(e["as_of_year"] for e in train), max(e["as_of_year"] for e in train)],
        "eval_years_as_of": [min(e["as_of_year"] for e in eval_), max(e["as_of_year"] for e in eval_)],
        "train_examples": len(train),
        "eval_examples": len(eval_),
        "domains": list(DOMAINS),
        "target_features_by_domain": {d: list(FEATURES[i]) for i, d in enumerate(DOMAINS)},
        "jepa_loss_trace": losses,
        "overall": {
            "jepa_cosine": float(np.mean([r["jepa_cosine_to_actual"] for r in rows])),
            "ridge_cosine": float(np.mean([r["ridge_cosine_to_actual"] for r in rows])),
            "persistence_cosine": float(np.mean([r["persistence_cosine_to_actual"] for r in rows])),
            "jepa_regime_match_rate": float(np.mean([r["jepa_regime_match"] for r in rows])),
        },
        "by_domain": grouped(rows, "domain"),
        "by_target_year": grouped(rows, "target_year"),
        "notes": [
            "Curated ordinal PIT states, not externally measured series.",
            "Held-out period is 1931-1933 targets, so collapse cases are outside the training target bank.",
            "Cosine is latent target-space alignment; regime-match uses nearest train target latent and is intentionally harsh for unseen collapse labels.",
        ],
    }

    with (out_dir / "weimar_states.jsonl").open("w", encoding="utf-8") as f:
        for s in states:
            f.write(json.dumps({"year": s.year, "matrix": s.matrix.tolist(), "events": s.events, "regime_label": s.regime_label}) + "\n")

    with (out_dir / "eval_rows.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/runs/weimar_graph_jepa_pilot"))
    args = parser.parse_args()
    summary = run_pilot(args.out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
