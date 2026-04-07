Local LLM proxy for VLviewer

This example proxy lets the browser call a local LLM (gpt4all/llama.cpp wrapper) without exposing remote API keys.

1) Install dependencies

```bash
cd server
npm install
```

2) Install Python and gpt4all (recommended approach)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install gpt4all
```

3) Download a gpt4all model (example)

Visit https://gpt4all.io/ or the official release repository and download a small quantized model such as `gpt4all-lora-quantized.bin`.
Place the model in `server/models/gpt4all-lora-quantized.bin` or set `GPT4ALL_MODEL_PATH` to its location.

4) Run the proxy using the Python wrapper (recommended)

```bash
# from project root
cd server
# ensure python env with gpt4all is active
# point MODEL_CMD to the wrapper script
export MODEL_CMD="$(pwd)/run_model.py"
node index.js
```

4.b) Alternative: use the JavaScript wrapper

If you prefer a JS wrapper that the proxy can invoke, set `MODEL_CMD` to `run_model_js.js`:

```bash
cd server
# make sure node deps are installed (from package.json in server)
export MODEL_CMD="$(pwd)/run_model_js.js"
node index.js
```

The JS wrapper will look for `GPT4ALL_BIN` or `MODEL_CMD` env vars (so you can point it to the real model CLI), or return simulated responses if none is configured.

5) Alternative: call the gpt4all CLI directly

If you have a gpt4all CLI binary that accepts a prompt as last argument or via stdin, set `MODEL_CMD` and optionally `MODEL_ARGS` in the environment before starting the proxy.

6) Start static server for the web app

```bash
# from project root
npx http-server -c-1 -p 8000 &
```

7) Test

Open http://localhost:8000, open the chat and write commands like "create box size 2 at 0 1 0". The proxy will call the wrapper which uses the local model.

Notes
- The wrapper prints a JSON object with `reply` and `actions`. If your model prints raw text, the wrapper packages it into that JSON. For production, secure access controls are required.
