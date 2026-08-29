"""Opt-in AI assist for hard images (glare, angle — Jenny, R9).

Off by default and hidden when no key is configured, so a deployment behind
Treasury's firewall (Marcus, R7) simply never calls out.
"""
import base64
import os

import anthropic

AI_MODEL = os.environ.get("AI_MODEL", "claude-opus-5")


class AIUnavailable(RuntimeError):
    pass


def ai_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def ai_extract_text(image_bytes: bytes, media_type: str) -> str:
    if not ai_available():
        raise AIUnavailable("ANTHROPIC_API_KEY not configured")
    data = base64.standard_b64encode(image_bytes).decode("utf-8")
    response = _client().with_options(timeout=30.0).messages.create(
        model=AI_MODEL,
        max_tokens=2048,
        output_config={"effort": "low"},  # transcription task; latency matters (R2)
        system=("Transcribe ALL text visible on this alcohol beverage label exactly as "
                "printed, preserving capitalization and line breaks. Output only the "
                "transcription."),
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
                                         "media_type": media_type, "data": data}},
            {"type": "text", "text": "Transcribe this label."},
        ]}],
    )
    return "".join(b.text for b in response.content if b.type == "text")
