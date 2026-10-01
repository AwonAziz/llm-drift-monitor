"""
Production shift simulator
--------------------------
Real drift is rarely a step function. This module produces a *schedule* of
windows, each with a named regime, a traffic mix and a text-style transform,
so the monitor faces a story it will also meet in production:

``baseline``    in-scope intents, original phrasing.
``volume_spike``same mix, 3x volume (tests that the detectors are sample-size
                aware and that nothing silently degrades under load).
``style_shift`` in-scope intents, but phrased by a new partner channel:
                lowercase, terse, emoji, no politeness. *Input drift with no
                label drift* — accuracy holds, the embedding space moves.
``new_intents`` out-of-scope intents (new products shipped). This is the
                dangerous one: the model cannot be right about an intent it was
                never trained on, and only quality monitoring catches it.
``mixed_crisis``out-of-scope traffic plus degraded phrasing plus a higher
                share of ambiguous/short queries.
``recovery``    after retraining: traffic normalises and the new intents are
                now in scope.

The transform functions are deterministic given a seed, so a demo run is
reproducible and a regression test can assert on exact behaviour.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from config import settings
from src.data.datasets import TEXT_COL
from src.utils.logging import get_logger
from src.utils.stats import stable_hash

logger = get_logger(__name__)

POLITE_PREFIX = (
    "Hi there, ", "Hello, ", "Good morning, ", "Hey, ", "Dear support team, ",
    "Hi, ", "Morning, ", "Hello support, ",
)
POLITE_SUFFIX = (" Thanks!", " Thank you.", " Appreciate your help.", " Please advise.",
                " Any update on this?", " Can someone check?", " Kind regards,")

TERSE_PREFIX = ("", "yo ", "hi ", "ok so ", "quick q - ", "urgent: ", "pls ", "asap ")
TERSE_SUFFIX = ("", " thx", " plz help", " asap", " fix pls", " 👀", " 🙏", "???")

NON_NATIVE_SUFFIX = (
    " plz", " is problem", " i dont know what to do", " can you help me thank you very much",
    " sorry for my english", " i am very worry about this", " pls advise urgent",
)

# Deliberate typos: adjacent-key swaps and dropped letters, the way humans
# actually type on a phone.
_TYPO_MAP = str.maketrans({
    "e": "r", "i": "o", "o": "i", "n": "m", "t": "r", "a": "s", "s": "d",
})


def _rng(seed: int, salt: str) -> np.random.Generator:
    return np.random.default_rng((stable_hash(salt) + seed) % (2**32))


# ── Text transforms ────────────────────────────────────────────────────

def transform_baseline(text: str, rng: np.random.Generator) -> str:
    return text


def transform_polished(text: str, rng: np.random.Generator) -> str:
    """Slightly more formal than the raw dataset — the original web channel."""
    if rng.random() < 0.55:
        return text
    return rng.choice(list(POLITE_PREFIX)) + text[0].lower() + text[1:] + rng.choice(list(POLITE_SUFFIX))


def transform_partner_channel(text: str, rng: np.random.Generator) -> str:
    """
    A new partner app routes traffic: terse, lowercase, no punctuation,
    occasionally an emoji. Semantics preserved — this is pure *style* drift.
    """
    out = text.strip().lower().rstrip("?.")
    if rng.random() < 0.45:
        out = rng.choice(list(TERSE_PREFIX)) + out
    if rng.random() < 0.35:
        out += rng.choice(list(TERSE_SUFFIX))
    if rng.random() < 0.12:
        out = out.replace("my ", "me ").replace("i am", "im").replace(" i ", " me ")
    if rng.random() < 0.08:
        words = out.split()
        if len(words) > 3:
            i = int(rng.integers(0, len(words) - 1))
            words[i] = words[i].translate(_TYPO_MAP)
            out = " ".join(words)
    return out


def transform_non_native(text: str, rng: np.random.Generator) -> str:
    """Non-native English phrasing — a classic post-expansion drift vector."""
    out = text.strip().rstrip(".")
    if rng.random() < 0.6:
        out = out[0].lower() + out[1:] if out else out
    if rng.random() < 0.7:
        out += rng.choice(list(NON_NATIVE_SUFFIX))
    if rng.random() < 0.25:
        words = out.split()
        if len(words) > 4:
            i = int(rng.integers(0, len(words) - 1))
            words[i] = words[i].translate(_TYPO_MAP)
            out = " ".join(words)
    return out


def transform_degraded(text: str, rng: np.random.Generator) -> str:
    """Crisis-window text: clipped, noisy, sometimes uninformative."""
    pick = rng.random()
    if pick < 0.18:
        words = text.split()
        return " ".join(words[: max(3, len(words) // 3)])      # truncated query
    if pick < 0.30:
        return text.lower()
    if pick < 0.42:
        return transform_non_native(text, rng)
    if pick < 0.52:
        return transform_partner_channel(text, rng)
    if pick < 0.58:
        words = text.split()
        return " ".join(w for w in words if rng.random() > 0.35) or "help"
    return transform_polished(text, rng)


TRANSFORMS: dict[str, Callable[[str, np.random.Generator], str]] = {
    "raw": transform_baseline,
    "polished": transform_polished,
    "partner_channel": transform_partner_channel,
    "non_native": transform_non_native,
    "degraded": transform_degraded,
}


# ── Regimes ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Regime:
    name: str
    n_windows: int
    oos_share: float                       # share of out-of-scope intents
    transform: str
    transform_mix: tuple[str, ...] = ()    # blend multiple channel styles
    weight_mult: float = 1.0
    label_noise: float = 0.0               # corrupted gold labels (annotation drift)
    description: str = ""


def default_schedule() -> list[Regime]:
    """
    The headline demo storyline. Fourteen windows: three calm, then the shift
    lands in stages, then recovery after the retrain.
    """
    return [
        Regime("baseline", 3, 0.0, "polished", description="Retail card/ATM traffic as launched."),
        Regime("volume_spike", 1, 0.0, "polished", weight_mult=2.0,
               description="Marketing campaign floods the queue; mix unchanged."),
        Regime("style_shift", 3, 0.0, "partner_channel",
               description="New partner channel: terse, lowercase, emoji. Same intents."),
        Regime("new_intents", 3, 0.45, "polished",
               description="Remittance + virtual-card products ship. 45% out-of-scope traffic."),
        Regime("mixed_crisis", 2, 0.60, "degraded",
               label_noise=0.05,
               description="Out-of-scope traffic plus degraded phrasing plus label noise."),
        Regime("recovery", 2, 0.45, "polished",
               description="Post-retrain: new intents are now in scope and labelled."),
    ]


# ── Traffic generation ─────────────────────────────────────────────────

@dataclass
class TrafficWindow:
    frame: pd.DataFrame
    regime: Regime
    window_index: int
    label: str
    notes: str = ""
    expected: dict[str, Any] = field(default_factory=dict)


class TrafficSimulator:
    """Generates the production windows consumed by the monitor."""

    def __init__(self, df: pd.DataFrame, seed: int = 4242,
                 base_window_size: int = settings.WINDOW_SIZE,
                 label_coverage: float = 0.55,
                 train_pool: pd.DataFrame | None = None):
        self.full = df.reset_index(drop=True)
        self.seed = seed
        self.base_window_size = base_window_size
        self.label_coverage = label_coverage

        available = set(self.full["category"])
        launch = [i for i in settings.LAUNCH_INTENTS if i in available]
        oos = [i for i in settings.OUT_OF_SCOPE_INTENTS if i in available]
        if not launch:
            raise RuntimeError("No launch intents in the dataset.")
        missing = [i for i in settings.LAUNCH_INTENTS if i not in available]
        if missing:
            logger.warning("Ignoring %d unknown launch intents: %s", len(missing), missing[:4])
        missing_oos = [i for i in settings.OUT_OF_SCOPE_INTENTS if i not in available]
        if missing_oos:
            logger.warning("Ignoring %d unknown OOS intents: %s", len(missing_oos), missing_oos[:4])

        self.launch = launch
        self.oos = oos
        self.pools = {
            intent: self.full.loc[self.full["category"] == intent, TEXT_COL].tolist()
            for intent in sorted(set(launch) | set(oos))
        }
        self.pools = {k: v for k, v in self.pools.items() if len(v) >= 2}
        logger.info("Simulator pools: %d in-scope intents, %d out-of-scope intents",
                    len([i for i in self.pools if i in set(launch)]),
                    len([i for i in self.pools if i in set(oos)]))

    # ── sampling ────────────────────────────────────────────────────
    def _sample_intent(self, intent: str, rng: np.random.Generator) -> str:
        pool = self.pools[intent]
        return str(pool[int(rng.integers(0, len(pool)))])

    def generate_window(self, regime: Regime, window_index: int,
                        n: int | None = None, in_scope_intents: Sequence[str] | None = None) -> TrafficWindow:
        rng = _rng(self.seed, f"{regime.name}:{window_index}")
        size = int((n or self.base_window_size) * regime.weight_mult)
        in_scope_intents = list(in_scope_intents or self.launch)
        oos_intents = self.oos or in_scope_intents

        n_oos = int(round(size * regime.oos_share))
        n_in = size - n_oos

        rows: list[dict[str, Any]] = []
        plan: list[tuple[str, int, bool]] = []
        if n_in > 0 and in_scope_intents:
            share = max(1, n_in // len(in_scope_intents))
            plan += [(i, share, True) for i in in_scope_intents]
        if n_oos > 0 and oos_intents:
            share = max(1, n_oos // len(oos_intents))
            plan += [(i, share, False) for i in oos_intents]

        for intent, count, scope in plan:
            if intent not in self.pools or count <= 0:
                continue
            rows.extend(
                {"text": self._sample_intent(intent, rng), "gold_intent": intent, "in_scope": scope}
                for _ in range(count)
            )
        if not rows:
            raise RuntimeError(f"Regime {regime.name} produced no traffic.")
        frame = pd.DataFrame(rows).head(size).reset_index(drop=True)

        styles = regime.transform_mix or (regime.transform,)
        weights = _normalised(np.linspace(1.0, 0.4, len(styles)))
        frame["text"] = [
            TRANSFORMS[str(rng.choice(styles, p=weights))](t, rng)
            for t in frame["text"]
        ]

        # Annotation drift: a real backlog of support ops gets some labels wrong.
        if regime.label_noise > 0:
            n_corrupt = int(len(frame) * regime.label_noise)
            if n_corrupt:
                idx = rng.choice(len(frame), size=n_corrupt, replace=False)
                pool = [i for i in sorted(frame["gold_intent"].unique())]
                for i in idx:
                    current = frame.at[i, "gold_intent"]
                    alt = [p for p in pool if p != current]
                    if alt:
                        frame.at[i, "gold_intent"] = str(rng.choice(alt))

        # Ground truth only exists where support ops resolved the ticket.
        covered = rng.random(len(frame)) < self.label_coverage
        frame["labeled"] = covered
        frame.loc[~covered, "gold_intent"] = None
        frame["channel"] = str(rng.choice(styles))
        frame["text_length"] = frame["text"].str.len()

        expected = {
            "oos_share": regime.oos_share,
            "style": regime.transform,
            "labels_expected": regime.label_noise > 0,
        }
        return TrafficWindow(
            frame=frame, regime=regime, window_index=window_index,
            label=f"{regime.name}", notes=regime.description, expected=expected,
        )

    def iter_schedule(self, schedule: Sequence[Regime] | None = None,
                      n: int | None = None,
                      in_scope_intents: Sequence[str] | None = None):
        schedule = list(schedule or default_schedule())
        idx = 0
        for regime in schedule:
            for _ in range(regime.n_windows):
                yield self.generate_window(regime, idx, n=n, in_scope_intents=in_scope_intents)
                idx += 1


def _normalised(weights: np.ndarray) -> np.ndarray:
    total = weights.sum()
    return weights / total if total > 0 else np.full(weights.shape, 1.0 / weights.size)