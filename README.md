# OpenAI-Compatible Server

An extensible FastAPI Chat Completions server with multimodal messages, reasoning
content, OpenAI tool fields, vLLM-style sampling parameters, SSE streaming, and
pluggable model backends.

## Setup

```powershell
uv sync
uv run openai-compatible-server
```

The API is available at `http://127.0.0.1:8000`, with interactive docs at
`http://127.0.0.1:8000/docs`.

## Commands

Start the server:

```powershell
uv run openai-compatible-server
```

Call it with a generic HTTP client:

```powershell
uv run openai-compatible-http-client `
  --client httpx-async `
  --url http://127.0.0.1:8000/v1/chat/completions `
  --model demo-multimodal-model `
  --prompt "Describe the input." `
  --stream
```

Call it with the OpenAI SDK:

```powershell
uv run openai-compatible-openai-sdk-client `
  --base-url http://127.0.0.1:8000/v1 `
  --model demo-multimodal-model `
  --prompt "Describe the input." `
  --stream
```

Both clients support common generation and multimodal options:

```powershell
uv run openai-compatible-openai-sdk-client `
  --api-key "your-api-key" `
  --model "my-model" `
  --prompt "What is in this image?" `
  --image ".\example.png" `
  --video "https://example.com/example.mp4" `
  --timeout 120 `
  --max-tokens 512 `
  --temperature 0.7 `
  --top-p 0.9 `
  --top-k 40 `
  --min-p 0.05 `
  --repetition-penalty 1.05 `
  --frequency-penalty 0.0 `
  --presence-penalty 0.0 `
  --min-tokens 1 `
  --thinking-token-budget 256 `
  --reasoning-effort medium `
  --seed 42 `
  -n 1 `
  --stream
```

Show all client options:

```powershell
uv run openai-compatible-http-client --help
uv run openai-compatible-openai-sdk-client --help
```

Development and packaging commands:

```powershell
uv run openai-compatible-build-wheel
uv run pytest
uv run ruff check .
```

## Build Wheel

Build a clean wheel into `dist/`:

```powershell
uv run openai-compatible-build-wheel
```

Use another output directory or preserve existing wheels:

```powershell
uv run openai-compatible-build-wheel --out-dir artifacts
uv run openai-compatible-build-wheel --no-clean
```

The equivalent native uv command is:

```powershell
uv build --wheel
```

Install the generated package:

```powershell
uv pip install dist/openai_compatible_server-0.1.0-py3-none-any.whl
```

## Run Without uv

After building the wheel, it can be installed and run with standard Python
tools. `uv` is not required on the target computer:

```bash
python -m pip install dist/openai_compatible_server-0.1.0-py3-none-any.whl
openai-compatible-server
```

You can also start the server through its Python module. This is useful when
the Python `bin` or `Scripts` directory is not available in `PATH`:

```bash
python -m openai_compatible
```

Configure and start the installed server with Bash or Zsh:

```bash
export HOST="0.0.0.0"
export PORT="8000"
export MODEL_ID="my-model"
python -m openai_compatible
```

Fish:

```fish
set -x HOST "0.0.0.0"
set -x PORT "8000"
set -x MODEL_ID "my-model"
python -m openai_compatible
```

PowerShell:

```powershell
$env:HOST = "0.0.0.0"
$env:PORT = "8000"
$env:MODEL_ID = "my-model"
python -m openai_compatible
```

Windows Command Prompt:

```bat
set HOST=0.0.0.0
set PORT=8000
set MODEL_ID=my-model
python -m openai_compatible
```

## Custom Model

Subclass `BaseModelBackend` and implement `load_model()` and `infer()`. The
default `stream_generate()` converts complete results to SSE chunks; override it
when the model supports native token streaming.

See `examples/custom_backend.py`, then configure:

PowerShell:

```powershell
$env:MODEL_BACKEND_CLASS = "examples.custom_backend:CustomModelBackend"
$env:MODEL_ID = "my-model"
$env:MODEL_MAX_CONCURRENCY = "2"
uv run openai-compatible-server
```

Bash or Zsh:

```bash
export MODEL_BACKEND_CLASS="examples.custom_backend:CustomModelBackend"
export MODEL_ID="my-model"
export MODEL_MAX_CONCURRENCY="2"
uv run openai-compatible-server
```

Fish:

```fish
set -x MODEL_BACKEND_CLASS "examples.custom_backend:CustomModelBackend"
set -x MODEL_ID "my-model"
set -x MODEL_MAX_CONCURRENCY "2"
uv run openai-compatible-server
```

Windows Command Prompt:

```bat
set MODEL_BACKEND_CLASS=examples.custom_backend:CustomModelBackend
set MODEL_ID=my-model
set MODEL_MAX_CONCURRENCY=2
uv run openai-compatible-server
```

## Environment

| Variable | Default | Purpose |
| --- | --- | --- |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `API_KEY` | unset | Optional Bearer authentication |
| `MODEL_ID` | `demo-multimodal-model` | Advertised model ID |
| `MODEL_BACKEND_CLASS` | unset | Backend as `module:ClassName` |
| `MODEL_MAX_CONCURRENCY` | `1` | Concurrent backend inference calls |
| `MODEL_STREAM_CHUNK_SIZE` | `12` | Default stream adapter chunk size |
| `LOG_DIR` | `logs` | Log directory |
| `LOG_LEVEL` | `INFO` | Log level |
| `LOG_ROTATION` | `00:00` | Loguru rotation policy |
| `LOG_RETENTION` | `14 days` | Log retention policy |

Video input uses the compatibility extensions `video_url` and `input_video`.
OpenAI Chat Completions does not define a standard video content part.
