# Python Examples

This folder contains eleven standalone Python scripts that walk through the LiteRT-LM Python API step by step. Each script imports only the packages it needs and is short enough to read in a couple of minutes.

---

## Setup

Install the dependencies from the project root:

```bash
pip install -r ../requirements.txt
```

Or with `uv`:

```bash
uv pip install -r ../requirements.txt
```

---

## Getting the model

Scripts 01 through 10 load the model from a file on disk. The file must be in **the same directory where you run the script** (i.e. this `python/` folder after you move the scripts here, or the repo root if you run from there).

**Download with curl:**

```bash
curl -L \
  "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm?download=true" \
  -o gemma-4-E2B-it.litertlm
```

**Or open the link directly in your browser:**

```
https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm?download=true
```

The file is a few gigabytes. After the download finishes, all scripts referencing `MODEL_PATH = "gemma-4-E2B-it.litertlm"` will find it automatically.

Script 11 is different. It talks to a running local server, and that server looks up models from the LiteRT-LM local store rather than from a bare file path. See the section on example 11 below for those setup steps.

---

## Running an example

```bash
python 01_basic_chat.py
```

Every script suppresses verbose logs with `litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)` so your terminal stays readable.

---

## Script-by-script guide

### 01_basic_chat.py

The most minimal example. Opens an engine, creates a conversation, sends one message, and prints the full response text. A good starting point if you want to confirm the setup is working.

```bash
python 01_basic_chat.py
```

### 02_streaming_chat_loop.py

An interactive chat loop that streams tokens to the terminal as they are generated. Type anything and press Enter to get a response. Type `exit` or `quit` to stop. This is the closest thing to a chat application in the cookbook.

```bash
python 02_streaming_chat_loop.py
```

### 03_system_prompt.py

Shows how to seed the conversation with a system message before the first user turn. In this example the model is told to act as a concise Python expert. Two questions are asked back-to-back to show that the persona persists across turns.

```bash
python 03_system_prompt.py
```

### 04_gpu_backend.py

Same as example 01 but the engine is initialised with `backend=litert_lm.Backend.GPU`. A `cache_dir` is also specified so compiled GPU kernels are stored between runs and do not need to be recompiled every time. Requires a compatible GPU and the GPU-enabled build of LiteRT-LM.

```bash
python 04_gpu_backend.py
```

### 05_speculative_decoding.py

Adds `enable_speculative_decoding=True` on top of the GPU backend. Multi-token prediction lets the model draft several tokens at once and verify them in parallel, which can noticeably reduce the time to first token and overall generation time. Requires GPU.

```bash
python 05_speculative_decoding.py
```

### 06_tool_use.py

Defines three Python functions (`add_numbers`, `multiply_numbers`, `get_current_weather`) and passes them to the conversation as tools. When the model decides a question requires calculation or a data lookup it emits a tool call, the library executes the corresponding function, and the result is fed back automatically.

```bash
python 06_tool_use.py
```

### 07_multimodal_audio.py

Demonstrates how to send an audio file together with a text prompt. The user message is a list with an `audio` content block pointing to a `.wav` file and a `text` block with the instruction. Update `AUDIO_FILE` to point to a real audio file before running. Note that audio support depends on the specific model variant being used.

```bash
python 07_multimodal_audio.py
```

### 08_multimodal_vision.py

Same idea as example 07, but with an image. The user message contains an `image` content block and a `text` block asking the model to describe what it sees. Update `IMAGE_FILE` to point to a real image before running. Vision support also depends on the model variant.

```bash
python 08_multimodal_vision.py
```

### 09_streaming_with_system_prompt.py

Combines a system persona (a senior software engineer specialising in on-device AI) with streaming output. The response is printed token by token. A good template for applications that need both a custom persona and a responsive interface.

```bash
python 09_streaming_with_system_prompt.py
```

### 10_all_features.py

A kitchen-sink demo that switches on GPU, speculative decoding, tools, and streaming all at once. Three queries are sent sequentially. Two of them exercise tools (temperature conversion and word counting) and one is a free-form question. A good reference for a production-like setup.

```bash
python 10_all_features.py
```

### 11_openai_api_server.py

This script does not load a model file directly. Instead it connects to a local LiteRT-LM server and sends requests over HTTP. There are two server modes and you need to pick one before running the script.

**Step 1: import the model into the local store (one-time)**

```bash
litert-lm import \
    --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm \
    gemma-4-E2B-it.litertlm
```

The model ends up at `~/.litert-lm/models/gemma-4-E2B-it.litertlm/model.litertlm`. This step only needs to run once.

**Step 2a: start the OpenAI Responses API server**

```bash
litert-lm serve --api openai --host localhost --port 9379
```

**Step 2b: or start the Gemini API server**

```bash
litert-lm serve --api gemini --host localhost --port 9379
```

**Step 3: run the client**

```bash
python 11_openai_api_server.py
```

The script contains four sections under the OpenAI API (`A1` through `A4`) and four under the Gemini API (`B1` through `B4`). The Gemini ones are commented out by default because only one server can be active at a time. Uncomment the ones matching your chosen server mode.

OpenAI API patterns covered:
- `A1` plain SDK call
- `A2` streaming via raw SSE
- `A3` raw HTTP POST without the SDK
- `A4` raw HTTP streaming

Gemini API patterns covered:
- `B1` basic generateContent
- `B2` generateContent with a system instruction
- `B3` multi-turn conversation (history passed in the request body)
- `B4` streaming via streamGenerateContent

---

## Notes on GPU examples

GPU examples (04, 05, 10) require a GPU that LiteRT-LM supports. On first run the engine compiles and caches GPU kernels, so startup takes longer than usual. Subsequent runs reuse the cache from `cache_dir` and start up faster.

If you do not have a GPU, change `backend=litert_lm.Backend.GPU` to `backend=litert_lm.Backend.CPU` and remove `enable_speculative_decoding=True`. Everything else works the same way on CPU.
