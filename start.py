"""
RouteLLM launcher — starts Gemini proxy then RouteLLM server.

  Strong model: Gemini CLI via local proxy on :8080
  Weak model:   Ollama phi4 on :11434 (change WEAK_MODEL to swap)
  Router:       mf (matrix factorization, best accuracy)
  RouteLLM:     OpenAI-compatible server on :6060

Usage:
    python start.py                   # defaults
    python start.py --threshold 0.2   # more Gemini, less Ollama
    python start.py --weak phi4       # pick a different Ollama model
"""

import argparse
import os
import subprocess
import sys
import time
import urllib.request

# Use the hermes venv Python which has routellm + fastapi installed
HERMES_PYTHON = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    "hermes", "hermes-agent", "venv", "Scripts", "python.exe",
)
VENV_PYTHON = HERMES_PYTHON if os.path.exists(HERMES_PYTHON) else sys.executable

# ── tunables ────────────────────────────────────────────────────────────────
PROXY_PORT = 8080
ROUTELLM_PORT = 6060
STRONG_MODEL = "openai/gemini-cli"          # resolved via OPENAI_API_BASE below
WEAK_MODEL_DEFAULT = "ollama_chat/phi4"
ROUTER = "mf"
THRESHOLD = 0.11593  # ~50 % strong-model calls; lower = more Gemini
# ────────────────────────────────────────────────────────────────────────────


def wait_for_http(url: str, timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help="Cost threshold (lower = more strong-model calls)")
    parser.add_argument("--weak", default=WEAK_MODEL_DEFAULT,
                        help="Weak (cheap) model in LiteLLM format")
    parser.add_argument("--port", type=int, default=ROUTELLM_PORT,
                        help="RouteLLM server port")
    args = parser.parse_args()

    # Gemini proxy must resolve via OpenAI provider in LiteLLM
    os.environ.setdefault("OPENAI_API_KEY", "no-key")
    os.environ["OPENAI_API_BASE"] = f"http://127.0.0.1:{PROXY_PORT}/v1"

    here = os.path.dirname(os.path.abspath(__file__))

    # ── 1. Start Gemini CLI proxy ─────────────────────────────────────────
    print(f"Starting Gemini CLI proxy on port {PROXY_PORT}...")
    proxy = subprocess.Popen(
        [VENV_PYTHON, os.path.join(here, "gemini_proxy.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if not wait_for_http(f"http://127.0.0.1:{PROXY_PORT}/health", timeout=15):
        proxy.kill()
        sys.exit("ERROR: Gemini proxy failed to start.")
    print(f"  Gemini proxy ready at http://127.0.0.1:{PROXY_PORT}/v1")

    # ── 2. Start RouteLLM OpenAI-compatible server ────────────────────────
    print(f"Starting RouteLLM on port {args.port} (router={ROUTER}, threshold={args.threshold})...")
    config_path = os.path.join(here, "config.example.yaml")
    routellm_cmd = [
        VENV_PYTHON, "-m", "routellm.openai_server",
        "--routers", ROUTER,
        "--strong-model", STRONG_MODEL,
        "--weak-model", args.weak,
        "--config", config_path,
        "--port", str(args.port),
    ]
    server = subprocess.Popen(routellm_cmd)

    print(f"""
╔══════════════════════════════════════════════════════╗
║  RouteLLM is running                                 ║
╠══════════════════════════════════════════════════════╣
║  Endpoint:     http://localhost:{args.port}/v1          ║
║  Strong model: Gemini CLI (via proxy :{PROXY_PORT})  ║
║  Weak model:   {args.weak:<36} ║
║  Router:       {ROUTER} (threshold={args.threshold})                ║
╠══════════════════════════════════════════════════════╣
║  Use model string in your client:                    ║
║    router-{ROUTER}-{args.threshold}                         ║
╚══════════════════════════════════════════════════════╝
Press Ctrl+C to stop.
""")

    try:
        server.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.terminate()
        proxy.terminate()


if __name__ == "__main__":
    main()
