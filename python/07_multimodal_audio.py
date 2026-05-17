"""
Example 07: Multimodal input — Audio
Pass an audio file alongside a text prompt.

A short test WAV file is generated programmatically if AUDIO_FILE does not
exist, so the script runs out of the box without any external files.

Note: audio support requires a model built with audio encoder signatures.
The standard gemma-4-E4B-it.litertlm does not include an audio encoder,
so this example will raise a clear error explaining the limitation. Use a
multimodal variant when one becomes available in the litert-community repo.
"""

import math
import struct
import wave
from pathlib import Path

import litert_lm

MODEL_PATH = "gemma-4-E4B-it.litertlm"
AUDIO_FILE = "test_audio.wav"

litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)


def _generate_test_wav(path: str, duration_s: float = 1.0, frequency_hz: float = 440.0) -> None:
    """Write a mono 16-bit PCM WAV with a sine wave at the given frequency."""
    sample_rate = 16000
    n_samples = int(sample_rate * duration_s)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        for i in range(n_samples):
            value = int(32767 * math.sin(2 * math.pi * frequency_hz * i / sample_rate))
            wf.writeframes(struct.pack("<h", value))


if not Path(AUDIO_FILE).exists():
    print(f"Generating test WAV file: {AUDIO_FILE}")
    _generate_test_wav(AUDIO_FILE)

try:
    with litert_lm.Engine(
        MODEL_PATH,
        audio_backend=litert_lm.Backend.CPU,
    ) as engine:
        with engine.create_conversation() as conversation:
            user_message = {
                "role": "user",
                "content": [
                    {"type": "audio", "path": AUDIO_FILE},
                    {"type": "text", "text": "Transcribe and summarise this audio."},
                ],
            }
            response = conversation.send_message(user_message)
            print(response["content"][0]["text"])
except Exception as e:
    print(f"Audio inference failed: {e}")
    print(
        "\nNote: the standard gemma-4-E4B-it.litertlm model does not include an audio encoder. "
        "This example requires a model built with audio support. "
        "Check the litert-community Hugging Face repo for a multimodal variant."
    )
