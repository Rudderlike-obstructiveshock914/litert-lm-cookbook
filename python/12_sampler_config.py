"""
Example 12: SamplerConfig
Control generation behaviour — temperature, top_k, top_p, and seed —
to produce creative, conservative, or fully reproducible outputs.
"""

import litert_lm

MODEL_PATH = "gemma-4-E4B-it.litertlm"

litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

PROMPT = "Write a two-sentence story about a robot discovering music."

configs = [
    ("Deterministic (temp=0.0, seed=42)", litert_lm.SamplerConfig(temperature=0.0, seed=42)),
    ("Conservative (top_k=10, temp=0.3)", litert_lm.SamplerConfig(top_k=10, temperature=0.3)),
    ("Balanced (top_p=0.9, temp=0.8)", litert_lm.SamplerConfig(top_p=0.9, temperature=0.8)),
    ("Creative (top_p=0.95, temp=1.2)", litert_lm.SamplerConfig(top_p=0.95, temperature=1.2)),
]

with litert_lm.Engine(MODEL_PATH) as engine:
    for label, sampler_config in configs:
        print(f"--- {label} ---")
        with engine.create_conversation(sampler_config=sampler_config) as conversation:
            response = conversation.send_message(PROMPT)
            print(response["content"][0]["text"])
        print()
