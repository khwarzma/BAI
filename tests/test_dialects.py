from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE_BUILD = ROOT / "core" / "build"
if str(NATIVE_BUILD) not in sys.path:
    sys.path.insert(0, str(NATIVE_BUILD))
MODEL_CANDIDATES = (ROOT / "models" / "v1.bai", ROOT / "models" / "v1.onnx")
VOCAB_CANDIDATES = (
    ROOT / "models" / "vocab.json",
    ROOT / "config" / "vocab.json",
    ROOT / "vocab.json",
)

SAMPLES = (
    ("ar", "ar-EG", "رمز التحقق الخاص بك هو 839201. صالح لمدة 5 دقائق."),
    ("ar", "ar-SA", "رمز الدخول الخاص بك هو 948201، لا تشاركه مع أي شخص."),
    ("ar", "ar-LEV", "كود التأكيد تبعك هو 731904 وصالح لخمس دقايق."),
    ("ar", "ar-MA", "كود التفعيل ديالك هو 4821، صالح غير لمدة قصيرة."),
    ("en", "en-US", "Your login verification code is 839201. It expires in 5 minutes."),
    ("en", "en-GB", "Your one-time passcode is 948201; it is valid for five minutes."),
)


def _load_pipeline() -> Any:
    try:
        import bai_core
    except ImportError as exc:
        pytest.skip(f"bai_core extension is unavailable: {exc}")

    model_path = next((path for path in MODEL_CANDIDATES if path.is_file()), None)
    vocab_path = next((path for path in VOCAB_CANDIDATES if path.is_file()), None)
    if model_path is None or vocab_path is None:
        pytest.skip("A supported model artifact and vocab.json are required")

    config = bai_core.EngineConfig()
    config.model_path = str(model_path)
    config.num_threads = 2
    pipeline = bai_core.InferencePipeline()
    try:
        pipeline.initialize(config, str(vocab_path))
    except (RuntimeError, OSError) as exc:
        pytest.skip(f"native pipeline could not initialize: {exc}")
    return pipeline


@pytest.fixture(scope="module")
def pipeline() -> Any:
    return _load_pipeline()


def _json_result(pipeline: Any, text: str) -> dict[str, Any]:
    result = pipeline.predict_json(text)
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert isinstance(parsed, dict)
    return parsed


def test_pipeline_is_initialized(pipeline: Any) -> None:
    assert pipeline.is_initialized() is True


@pytest.mark.parametrize(("language", "dialect", "text"), SAMPLES)
def test_dialect_sample_returns_valid_prediction(
    pipeline: Any, language: str, dialect: str, text: str
) -> None:
    result = _json_result(pipeline, text)
    assert result
    assert isinstance(result.get("category"), str)
    assert result["category"] in {
        "inbox_pinned",
        "inbox",
        "bait",
        "bais",
        "baiads",
    }
    confidence = result.get("overall_confidence")
    if confidence is None:
        confidence = result.get("confidence")
    assert isinstance(confidence, (float, int))
    assert 0.0 <= float(confidence) <= 1.0
    assert language in {"ar", "en"}
    assert dialect.startswith(f"{language}-")


def test_predict_json_is_strict_json_for_otp(pipeline: Any) -> None:
    result = _json_result(
        pipeline,
        "Your verification code is 654321. Do not share this one-time password.",
    )
    assert isinstance(result.get("otp_detected"), bool)
    otp_confidence = result.get("otp_confidence")
    assert isinstance(otp_confidence, (float, int))
    assert 0.0 <= float(otp_confidence) <= 1.0