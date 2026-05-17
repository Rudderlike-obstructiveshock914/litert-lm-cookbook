"""
Example 08: Multimodal input — Vision (image)
Pass an image file alongside a text prompt.

A small test image is generated programmatically if IMAGE_FILE does not
exist, so the script runs out of the box without any external files.

Note: vision support requires a model with exactly one vision encoder
signature. The standard gemma-4-E4B-it.litertlm has three vision signatures
(vision_70, vision_140, vision_280), which the current LiteRT-LM runtime
does not support. This example will raise a clear error explaining the
limitation. Check the litert-community Hugging Face repo for a single-
signature vision model when one becomes available.
"""

import struct
import zlib
from pathlib import Path

import litert_lm

MODEL_PATH = "gemma-4-E4B-it.litertlm"
IMAGE_FILE = "test_image.png"

litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)


def _generate_test_png(path: str, width: int = 64, height: int = 64) -> None:
    """Write a minimal valid RGB PNG with a gradient pattern."""

    def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        return length + chunk_type + data + crc

    # Build raw pixel rows: filter byte 0x00 + RGB pixels
    raw_rows = b""
    for y in range(height):
        raw_rows += b"\x00"
        for x in range(width):
            r = int(255 * x / (width - 1))
            g = int(255 * y / (height - 1))
            b = 128
            raw_rows += bytes([r, g, b])

    compressed = zlib.compress(raw_rows)
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr_data)
        + _png_chunk(b"IDAT", compressed)
        + _png_chunk(b"IEND", b"")
    )

    Path(path).write_bytes(png)


if not Path(IMAGE_FILE).exists():
    print(f"Generating test PNG file: {IMAGE_FILE}")
    _generate_test_png(IMAGE_FILE)

try:
    with litert_lm.Engine(
        MODEL_PATH,
        vision_backend=litert_lm.Backend.CPU,
    ) as engine:
        with engine.create_conversation() as conversation:
            user_message = {
                "role": "user",
                "content": [
                    {"type": "image", "path": IMAGE_FILE},
                    {"type": "text", "text": "Describe what you see in this image."},
                ],
            }
            response = conversation.send_message(user_message)
            print(response["content"][0]["text"])
except Exception as e:
    print(f"Vision inference failed: {e}")
    print(
        "\nNote: the standard gemma-4-E4B-it.litertlm model has three vision encoder signatures "
        "(vision_70, vision_140, vision_280). The current LiteRT-LM runtime requires exactly one. "
        "This example will work once a single-signature vision model is available."
    )
