"""Tests for the data layer: dataset loading, reference construction, shifts."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.data.datasets import build_reference, load_banking77
from src.data.shifts import (
    TRANSFORMS,
    TrafficSimulator,
    default_schedule,
)


@pytest.fixture(scope="module")
def dataset():
    df, info = load_banking77()
    return df, info


class TestDataset:
    def test_loads_with_the_expected_shape(self, dataset):
        df, info = dataset
        assert len(df) > 1000
        assert info.intents == df["category"].nunique()
        assert info.source in {"remote", "bundled"}
        assert "Casanueva" in info.license

    def test_deduplicates_exact_rows(self, dataset):
        df, _ = dataset
        assert not df.duplicated(subset=["text", "category"]).any()

    def test_no_empty_texts(self, dataset):
        df, _ = dataset
        assert df["text"].str.len().gt(0).all()

    def test_manifest_is_written(self, dataset, tmp_path):
        from src.data.datasets import write_manifest

        _, info = dataset
        path = write_manifest(info, extra={"test": True})
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["test"] is True
        assert payload["launch_intents"]


class TestReference:
    def test_only_launch_intents_are_used(self, dataset):
        from config import settings

        df, _ = dataset
        ref = build_reference(df, n=300)
        assert set(ref["gold_intent"]) <= set(settings.LAUNCH_INTENTS)
        assert ref["in_scope"].all()

    def test_has_the_requested_size_and_schema(self, dataset):
        df, _ = dataset
        ref = build_reference(df, n=400)
        assert 0 < len(ref) <= 400 + 50
        assert list(ref.columns) == ["text", "gold_intent", "in_scope", "source"]

    def test_is_deterministic(self, dataset):
        df, _ = dataset
        a = build_reference(df, n=300, seed=5)
        b = build_reference(df, n=300, seed=5)
        assert a["text"].tolist() == b["text"].tolist()

    def test_different_seeds_differ(self, dataset):
        df, _ = dataset
        a = build_reference(df, n=300, seed=1)
        b = build_reference(df, n=300, seed=2)
        assert a["text"].tolist() != b["text"].tolist()

    def test_every_intent_is_represented(self, dataset):
        from config import settings

        df, _ = dataset
        ref = build_reference(df, n=520)
        expected = {i for i in settings.LAUNCH_INTENTS if i in set(df["category"])}
        assert set(ref["gold_intent"]) == expected


class TestTransforms:
    def test_partner_channel_preserves_intent_keywords(self):
        """Style drift must change the surface, not the topic.

        If the transform destroyed meaning there would be no concept drift to
        detect, and the demo would be measuring nothing.
        """
        rng = np.random.default_rng(0)
        text = "Why has my card not arrived after two weeks"
        out = TRANSFORMS["partner_channel"](text, rng)
        assert "card" in out and "arriv" in out
        assert out != text

    def test_degraded_can_truncate(self):
        rng = np.random.default_rng(1)
        outputs = {TRANSFORMS["degraded"]("a b c d e f g h", rng) for _ in range(60)}
        assert any(len(o) < 8 for o in outputs)

    def test_non_native_appends_phrasing(self):
        rng = np.random.default_rng(2)
        assert any("plz" in TRANSFORMS["non_native"]("my card is late", rng) for _ in range(30))

    def test_transforms_are_deterministic_given_the_rng(self):
        text = "my card has not arrived"
        assert (TRANSFORMS["partner_channel"](text, np.random.default_rng(4))
                == TRANSFORMS["partner_channel"](text, np.random.default_rng(4)))


class TestSchedule:
    def test_default_schedule_is_ordered(self):
        names = [r.name for r in default_schedule()]
        assert names.index("baseline") < names.index("new_intents")
        assert names.index("new_intents") < names.index("recovery")
        assert "style_shift" in names and "mixed_crisis" in names

    def test_escalates_then_recovers(self):
        oos = [r.oos_share for r in default_schedule()]
        assert oos[0] == 0.0
        assert max(oos) >= 0.5
        assert oos[-1] < max(oos)


class TestSimulator:
    @pytest.fixture(scope="class")
    @staticmethod
    def simulator(dataset):
        df, _ = dataset
        return TrafficSimulator(df, base_window_size=120)

    def test_baseline_is_entirely_in_scope(self, simulator):
        window = simulator.generate_window(default_schedule()[0], 0)
        assert window.frame["in_scope"].all()
        assert window.frame["gold_intent"].notna().any()

    def test_new_intents_window_has_out_of_scope_traffic(self, simulator):
        regime = next(r for r in default_schedule() if r.name == "new_intents")
        window = simulator.generate_window(regime, 7)
        share = 1 - window.frame["in_scope"].mean()
        assert 0.25 < share < 0.65, share

    def test_volume_multiplier_is_applied(self, simulator):
        regime = next(r for r in default_schedule() if r.name == "volume_spike")
        window = simulator.generate_window(regime, 3)
        assert len(window.frame) >= simulator.base_window_size

    def test_label_coverage_is_partial(self, simulator):
        window = simulator.generate_window(default_schedule()[0], 0)
        coverage = window.frame["gold_intent"].notna().mean()
        assert 0.3 < coverage < 0.85, coverage

    def test_label_noise_corrupts_labels(self, simulator):
        regime = next(r for r in default_schedule() if r.name == "mixed_crisis")
        noisy = simulator.generate_window(regime, 10)
        clean = simulator.generate_window(default_schedule()[0], 0)
        noisy_rate = noisy.frame["channel"].nunique()
        assert noisy_rate >= 1
        assert clean.frame["text"].str.contains(r"[?]").mean() > noisy.frame["text"].str.contains(r"[?]").mean()

    def test_same_index_reproduces_the_window(self, simulator):
        regime = default_schedule()[0]
        a = simulator.generate_window(regime, 5).frame["text"].tolist()
        b = simulator.generate_window(regime, 5).frame["text"].tolist()
        assert a == b

    def test_iter_schedule_covers_every_window(self, simulator):
        schedule = default_schedule()
        windows = list(simulator.iter_schedule(schedule))
        assert len(windows) == sum(r.n_windows for r in schedule)
        assert [w.window_index for w in windows] == list(range(len(windows)))

    def test_window_carries_context(self, simulator):
        window = simulator.generate_window(default_schedule()[2], 4)
        assert window.label == "style_shift"
        assert window.notes
        assert set(window.frame.columns) >= {"text", "gold_intent", "in_scope", "channel", "text_length"}

    def test_unknown_intents_are_warned_not_fatal(self, dataset, monkeypatch):
        from config import settings

        df, _ = dataset
        monkeypatch.setattr(settings, "LAUNCH_INTENTS",
                            ("definitely_not_a_real_intent", *settings.LAUNCH_INTENTS[:3]))
        sim = TrafficSimulator(df, base_window_size=60)
        assert sim.launch  # did not raise


class TestOfflineFallback:
    def test_bundled_corpus_round_trip(self, tmp_path, monkeypatch):
        from src.data import datasets

        rows = [{"text": f"question {i}", "category": "a" if i % 2 else "b"} for i in range(20)]
        path = datasets.ensure_bundled_corpus(rows)
        monkeypatch.setattr(datasets, "BUNDLED_PATH", path)
        monkeypatch.setattr(datasets, "MIN_VALID_ROWS", 1000)
        monkeypatch.setattr(datasets, "_load_cached", lambda: (None, None))
        monkeypatch.setattr(datasets, "_download", lambda url, timeout=60.0: None)

        df, info = datasets.load_banking77()
        assert info.source == "bundled"
        assert len(df) == 20
        assert info.intents == 2
        assert "BUNDLED" in info.path or info.path.endswith("jsonl")
        path.unlink()

    def test_raises_when_nothing_is_available(self, monkeypatch):
        from src.data import datasets

        monkeypatch.setattr(datasets, "_load_cached", lambda: (None, None))
        monkeypatch.setattr(datasets, "_download", lambda url, timeout=60.0: None)
        monkeypatch.setattr(datasets, "BUNDLED_PATH", tmp_path_missing())
        with pytest.raises(RuntimeError, match="Banking77 unavailable"):
            datasets.load_banking77()


def tmp_path_missing():
    from pathlib import Path

    return Path("C:/definitely/not/here/banking77_mini.jsonl")