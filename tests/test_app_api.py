"""Tests for the application under observation and the serving API."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.app.assistant import (
    PLAYBOOKS,
    AssistantOutput,
    IntentClassifier,
    SupportAssistant,
    humanise,
    templated_response,
)
from src.llm import build_llm


@pytest.fixture(scope="module")
def trained(tiny_dataset_module):
    return IntentClassifier.train(tiny_dataset_module, version="test-v1")


@pytest.fixture(scope="module")
def tiny_dataset_module():
    rng = np.random.default_rng(5)
    rows = []
    vocab = {
        "card_arrival": "card arrived delivery post tracking",
        "pin_blocked": "pin blocked locked forgot password",
        "atm_support": "atm cash machine withdrawal receipt",
        "top_up_limits": "top up limit balance deposit",
    }
    for intent, words in vocab.items():
        for k in range(45):
            extra = " ".join(rng.choice(words.split(), size=2))
            rows.append({"text": f"{extra} help number {k} {intent.replace('_', ' ')}",
                         "gold_intent": intent, "in_scope": True})
    for intent in ("virtual_card", "transfer_timing"):
        for k in range(30):
            rows.append({"text": f"{intent.replace('_', ' ')} question {k}",
                         "gold_intent": intent, "in_scope": False})
    return pd.DataFrame(rows).sample(frac=1.0, random_state=1).reset_index(drop=True)


@pytest.fixture()
def client(trained):
    from src.api import server as server_module

    server_module._state["classifier"] = trained
    server_module._state["requests"] = 0
    server_module._state["errors"] = 0
    return TestClient(server_module.app)


class TestClassifier:
    def test_learns_the_training_distribution(self, trained, tiny_dataset_module):
        preds = trained.predict_frame(tiny_dataset_module, "text")
        in_scope = tiny_dataset_module["in_scope"].to_numpy()
        acc = float((preds["pred_intent"].to_numpy()[in_scope]
                     == tiny_dataset_module["gold_intent"].to_numpy()[in_scope]).mean())
        assert acc > 0.85, acc

    def test_confidences_are_probabilities(self, trained):
        preds = trained.predict(["my card has not arrived", "totally unrelated text"])
        for p in preds:
            assert 0.0 <= p["confidence"] <= 1.0
            assert 0.0 <= p["margin"] <= 1.0
            assert len(p["top3"]) == 3

    def test_abstention_fires_on_low_confidence(self, trained):
        trained.abstain_threshold = 0.99
        preds = trained.predict(["card arrival delivery tracking"])
        assert all(p["abstained"] for p in preds)
        trained.abstain_threshold = 0.45

    def test_temperature_lowers_confidence_when_above_one(self, trained):
        base = trained.predict(["card arrival"])[0]["confidence"]
        trained.temperature = 3.0
        cooled = trained.predict(["card arrival"])[0]["confidence"]
        assert cooled < base
        trained.temperature = 1.0

    def test_round_trips_through_disk(self, trained, tmp_path):
        path = trained.save(tmp_path / "clf.joblib")
        loaded = IntentClassifier.load(path)
        assert loaded.version == trained.version
        assert loaded.classes == trained.classes
        assert loaded.temperature == trained.temperature
        assert loaded.predict(["card"])[0]["intent"] == trained.predict(["card"])[0]["intent"]

    def test_missing_artifact_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            IntentClassifier.load(tmp_path / "nope.joblib")

    def test_fit_temperature_shrinks_overconfidence(self, tiny_dataset_module):
        clf = IntentClassifier.train(tiny_dataset_module, version="t")
        t = clf.fit_temperature(tiny_dataset_module)
        assert t != 1.0
        assert 0.2 <= t <= 6.0

    def test_signature_includes_temperature(self, trained):
        assert "intents" in trained.signature and "T" in trained.signature


class TestAssistant:
    def test_produces_a_complete_output(self, trained):
        assistant = SupportAssistant(trained, llm=build_llm("mock"), use_llm=True)
        out = assistant.handle("my card has not arrived after two weeks")
        assert isinstance(out, AssistantOutput)
        assert out.intent in trained.classes
        assert out.response
        assert out.app_latency_ms > 0
        assert out.request_id.startswith("req_")

    def test_templated_path_works_without_an_llm(self, trained):
        assistant = SupportAssistant(trained, llm=build_llm("mock"), use_llm=False)
        out = assistant.handle("my card has not arrived")
        assert "card arrival" in out.response.lower()

    def test_llm_failure_degrades_to_a_template(self, trained):
        class Broken:
            provider = "broken"
            model = "broken"

            def generate(self, *a, **k):
                raise RuntimeError("upstream 503")

        assistant = SupportAssistant(trained, llm=Broken(), use_llm=True)
        out = assistant.handle("my card has not arrived")
        assert out.response

    def test_abstention_uses_the_default_playbook(self, trained):
        trained.abstain_threshold = 0.99
        assistant = SupportAssistant(trained, llm=build_llm("mock"), use_llm=False)
        out = assistant.handle("card arrival")
        trained.abstain_threshold = 0.45
        assert out.abstained is True

    def test_to_record_shape(self, trained):
        assistant = SupportAssistant(trained, llm=build_llm("mock"), use_llm=False)
        record = assistant.handle("card").to_record(gold_intent="card_arrival", in_scope=True)
        assert set(record) >= {"request_id", "text_hash", "pred_intent", "confidence",
                               "abstained", "response", "in_scope", "gold_intent"}

    def test_prompt_includes_the_playbook_and_confidence(self, trained):
        assistant = SupportAssistant(trained, llm=build_llm("mock"), use_llm=False)
        prompt = assistant.build_responder_prompt(
            "hi", "card_arrival", PLAYBOOKS["card_arrival"], 0.77)
        assert "3-5 working days" in prompt
        assert "0.77" in prompt

    def test_humanise(self):
        assert humanise("card_arrival") == "Card arrival"
        assert humanise("atm_support") == "Atm support"

    def test_templated_response_mentions_the_intent(self):
        out = templated_response("pin_blocked", PLAYBOOKS["pin_blocked"])
        assert "pin blocked" in out.lower()


class TestAPI:
    def test_health(self, client):
        body = client.get("/health").json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True
        assert body["intents"] > 0

    def test_predict(self, client):
        response = client.post("/predict", json={"text": "my card has not arrived",
                                                "generate_response": False})
        assert response.status_code == 200
        body = response.json()
        assert body["pred_intent"]
        assert 0 <= body["intent_confidence"] <= 1
        assert body["latency_ms"] >= 0
        assert body["request_id"].startswith("req_")

    def test_predict_rejects_empty_text(self, client):
        assert client.post("/predict", json={"text": ""}).status_code == 422

    def test_predict_rejects_an_unknown_forced_intent(self, client):
        response = client.post("/predict", json={"text": "hi", "intent": "not_an_intent"})
        assert response.status_code == 422

    def test_predict_survives_a_broken_llm(self, client):
        from src.api import server as server_module

        assistant = server_module._get_assistant()
        original = assistant.llm

        class Broken:
            provider, model = "broken", "broken"

            def generate(self, *a, **k):
                raise RuntimeError("down")

        try:
            assistant.llm = Broken()
            body = client.post("/predict", json={"text": "card not here"}).json()
            assert body["response"], "the serving path must degrade, not 500"
        finally:
            assistant.llm = original

    def test_batch_predict(self, client):
        payload = {"instances": [{"text": f"card question {i}"} for i in range(5)]}
        body = client.post("/predict/batch", json=payload).json()
        assert body["n"] == 5
        assert len(body["predictions"]) == 5

    def test_batch_size_limit(self, client):
        payload = {"instances": [{"text": "x"} for _ in range(201)]}
        assert client.post("/predict/batch", json=payload).status_code == 422

    def test_model_info(self, client):
        body = client.get("/model/info").json()
        assert body["intents"] > 0
        assert body["temperature"] > 0

    def test_monitoring_endpoints_answer(self, client):
        """The monitoring read model must never 500, whatever is in the store."""
        status = client.get("/monitoring/status")
        assert status.status_code == 200
        assert set(status.json()) >= {"run_id", "windows", "requests", "open_incidents"}
        assert client.get("/monitoring/runs").status_code == 200
        for path in ("/monitoring/decisions", "/monitoring/incidents"):
            assert client.get(path).status_code == 200
            assert isinstance(client.get(path).json(), list)

    def test_unknown_run_is_a_clean_404(self, client):
        for path in ("/monitoring/drift", "/monitoring/quality", "/monitoring/judge"):
            response = client.get(path, params={"run_id": "run_does_not_exist"})
            assert response.status_code == 404
            assert "No monitoring run" in response.json()["detail"]

    def test_metric_series_round_trip(self, client):
        runs = client.get("/monitoring/runs").json()
        if not runs:
            pytest.skip("no monitoring run recorded")
        run_id = runs[0]["run_id"]
        body = client.get("/monitoring/series", params={
            "run_id": run_id, "category": "embedding", "name": "domain_classifier_auc"}).json()
        assert body["run_id"] == run_id
        assert body["name"] == "domain_classifier_auc"
        for point in body["points"]:
            assert "window_index" in point and "value" in point

    def test_prometheus_exposition(self, client):
        text = client.get("/metrics/prometheus").text
        assert "ldm_http_requests_total" in text
        assert "ldm_uptime_seconds" in text

    def test_openapi_is_valid(self, client):
        assert client.get("/openapi.json").status_code == 200
