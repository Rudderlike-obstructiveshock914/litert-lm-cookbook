"""
Example 16: Benchmark
Use the built-in Benchmark class to measure init time, time to first
token, and prefill / decode throughput without writing a timer yourself.
"""

import litert_lm

MODEL_PATH = "gemma-4-E4B-it.litertlm"

litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

configs = [
    ("GPU  — 256 prefill / 128 decode", litert_lm.Backend.GPU, False, 256, 128),
    ("GPU  — 512 prefill / 256 decode", litert_lm.Backend.GPU, False, 512, 256),
    ("GPU + MTP — 256 prefill / 128 decode", litert_lm.Backend.GPU, True, 256, 128),
]

for label, backend, speculative, prefill, decode in configs:
    print(f"--- {label} ---")
    bench = litert_lm.Benchmark(
        MODEL_PATH,
        backend=backend,
        enable_speculative_decoding=speculative,
        prefill_tokens=prefill,
        decode_tokens=decode,
    )
    info = bench.run()
    print(f"  init time          : {info.init_time_in_second:.2f}s")
    print(f"  time to first token: {info.time_to_first_token_in_second:.3f}s")
    print(
        f"  prefill            : {info.last_prefill_token_count} tok  "
        f"@ {info.last_prefill_tokens_per_second:.1f} tok/s"
    )
    print(
        f"  decode             : {info.last_decode_token_count} tok  "
        f"@ {info.last_decode_tokens_per_second:.1f} tok/s"
    )
    print()
