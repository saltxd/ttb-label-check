from unittest.mock import MagicMock, patch

import app.ai as ai


def test_ai_available_false_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert ai.ai_available() is False


def test_ai_extract_raises_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    try:
        ai.ai_extract_text(b"x", "image/png")
        assert False, "expected AIUnavailable"
    except ai.AIUnavailable:
        pass


def test_ai_extract_uses_vision_block(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    fake = MagicMock()
    fake.content = [MagicMock(type="text", text="SUNSET ALE 5.9%")]
    with patch.object(ai, "_client") as c:
        c.return_value.with_options.return_value.messages.create.return_value = fake
        out = ai.ai_extract_text(b"png-bytes", "image/png")
    assert out == "SUNSET ALE 5.9%"
    kwargs = c.return_value.with_options.return_value.messages.create.call_args.kwargs
    blocks = kwargs["messages"][0]["content"]
    assert blocks[0]["type"] == "image"
    assert blocks[0]["source"]["type"] == "base64"
    assert blocks[0]["source"]["media_type"] == "image/png"
