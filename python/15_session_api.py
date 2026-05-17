"""
Example 15: Session API
Low-level access to the model via run_prefill and run_decode.
Also demonstrates run_text_scoring to rank candidate answers by
log-likelihood — useful for classification and evaluation tasks.

The Session API does not apply a chat template automatically, so prompts
must be formatted with the Gemma-4 instruct format:
    <start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n
"""

import litert_lm

MODEL_PATH = "gemma-4-E4B-it.litertlm"

litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

_BOS = "<start_of_turn>"
_EOS = "<end_of_turn>"


def _prompt(text: str) -> str:
    return f"{_BOS}user\n{text}{_EOS}\n{_BOS}model\n"


with litert_lm.Engine(MODEL_PATH) as engine:
    print("=== Part A: generation ===")
    with engine.create_session() as session:
        session.run_prefill([_prompt("What is the capital of Japan? Answer in one word.")])
        result = session.run_decode()
        print(f"Response: {result.texts[0].replace(_EOS, '').strip()}\n")

    print("=== Part B: streaming ===")
    with engine.create_session() as session:
        session.run_prefill([_prompt("Count from 1 to 5, one number per line.")])
        print("Response: ", end="", flush=True)
        # The EOS token arrives split across multiple chunks, so buffer and
        # print only the portion before the first <end_of_turn>.
        buf = ""
        printed = 0
        for chunk in session.run_decode_async():
            buf += chunk.texts[0]
            if _EOS in buf:
                print(buf[printed : buf.index(_EOS)], end="", flush=True)
                break
            # Hold back enough chars to catch a partial EOS at the boundary.
            safe = len(buf) - len(_EOS) + 1
            if safe > printed:
                print(buf[printed:safe], end="", flush=True)
                printed = safe
        print("\n")

    print("=== Part C: text scoring ===")
    context = "Paris is the capital of"
    candidates = [" France.", " Germany.", " Italy.", " Spain."]

    scored = []
    for candidate in candidates:
        with engine.create_session() as session:
            session.run_prefill([context])
            result = session.run_text_scoring([candidate])
            scored.append((candidate, result.scores[0]))

    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    print(f"Context: {context!r}")
    print("Candidate scores (higher = more likely):")
    for text, score in ranked:
        print(f"  {score:+.4f}  {text.strip()!r}")
