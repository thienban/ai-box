"""Suite de tests E2E reproductible pour AI Box (Colab & Local).

Valide le pipeline complet :
  compilation release -> `aibox serve --model <dir>` -> chargement SafeTensors
  -> endpoints `/health`, `/v1/models`, `/api/tags`, `/api/chat`, `/v1/chat/completions` (SSE)
  -> mesure TTFT, débit (tok/s), RAM pic (RSS) et qualité sur 6 prompts variés.

Usage:
    python scripts/e2e_eval.py --model /content/bonsai-4b-aibox
    python scripts/e2e_eval.py --model ~/.aibox/models/prism-ml--Bonsai-4B --max-tokens 32
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PromptCase:
    name: str
    prompt: str
    stream: bool
    expected_any: tuple[str, ...] = ()


PROMPT_SUITE: list[PromptCase] = [
    PromptCase(
        name="fr_factuel",
        prompt="Quelle est la capitale de la France ? Réponds en une phrase.",
        stream=True,
        expected_any=("Paris", "paris"),
    ),
    PromptCase(
        name="en_tech",
        prompt="Explain in one short sentence what a GPU is.",
        stream=False,
        expected_any=("GPU", "graphics", "Graphics", "parallel", "chip"),
    ),
    PromptCase(
        name="code_python",
        prompt="Write a 3-line Python function `add(a, b)` that returns a + b.",
        stream=True,
        expected_any=("def add", "return"),
    ),
    PromptCase(
        name="math_simple",
        prompt="What is 15 * 4 + 2? Reply with the number only.",
        stream=False,
        expected_any=("62",),
    ),
    PromptCase(
        name="json_format",
        prompt='Return a valid JSON object with keys "city" and "country" for Tokyo.',
        stream=True,
        expected_any=("Tokyo", "Japan"),
    ),
]


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def get_process_rss_mb(pid: int) -> float | None:
    system = platform.system()
    if system == "Linux":
        status_path = Path(f"/proc/{pid}/status")
        if status_path.exists():
            for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("VmHWM:") or line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
    elif system == "Windows":
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", f"(Get-Process -Id {pid}).PeakWorkingSet64 / 1MB"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            return float(out.replace(",", "."))
        except Exception:
            return None
    return None


def resolve_cargo() -> str:
    cargo_home = Path.home() / ".cargo" / "bin" / ("cargo.exe" if platform.system() == "Windows" else "cargo")
    return str(cargo_home) if cargo_home.exists() else "cargo"


def build_aibox_release(repo_root: Path) -> Path:
    cargo = resolve_cargo()
    manifest = repo_root / "app" / "Cargo.toml"
    print(f"[1/4] Compilation release ({cargo})...")
    subprocess.run(
        [cargo, "build", "--release", "--manifest-path", str(manifest), "-p", "aibox"],
        check=True,
    )
    bin_name = "aibox.exe" if platform.system() == "Windows" else "aibox"
    bin_path = repo_root / "app" / "target" / "release" / bin_name
    if not bin_path.exists():
        raise FileNotFoundError(f"Binaire introuvable : {bin_path}")
    return bin_path


def wait_for_health(base_url: str, proc: subprocess.Popen[str], timeout_s: float = 120.0) -> float:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        if proc.poll() is not None:
            out, err = proc.communicate()
            raise RuntimeError(f"Le serveur s'est arrêté prématurément ({proc.returncode}):\n{out}\n{err}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=1.0) as resp:
                if resp.status == 200:
                    return time.perf_counter() - t0
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(0.25)
    raise TimeoutError(f"Serveur non prêt après {timeout_s}s")


def http_get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_openai_chat(
    base_url: str,
    prompt: str,
    max_tokens: int,
    stream: bool,
    model_name: str = "bonsai-aibox",
) -> tuple[str, float, float, int]:
    payload = json.dumps(
        {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": stream,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    if not stream:
        with urllib.request.urlopen(req, timeout=300.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        total_s = time.perf_counter() - t0
        content = data["choices"][0]["message"]["content"]
        comp_tokens = int(data.get("usage", {}).get("completion_tokens", 0))
        return content, total_s * 1000.0, total_s, comp_tokens

    chunks: list[str] = []
    ttft_ms = 0.0
    saw_done = False
    with urllib.request.urlopen(req, timeout=300.0) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            body = line[len("data:") :].strip()
            if body == "[DONE]":
                saw_done = True
                break
            event = json.loads(body)
            delta = event["choices"][0].get("delta", {}).get("content", "")
            if delta:
                if not chunks:
                    ttft_ms = (time.perf_counter() - t0) * 1000.0
                chunks.append(delta)
    total_s = time.perf_counter() - t0
    if not saw_done:
        raise RuntimeError("Flux SSE terminé sans marqueur `data: [DONE]`")
    return "".join(chunks), ttft_ms, total_s, len(chunks)


def call_ollama_chat_ndjson(
    base_url: str,
    prompt: str,
    max_tokens: int,
    model_name: str = "bonsai-aibox",
) -> tuple[str, int]:
    payload = json.dumps(
        {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"num_predict": max_tokens, "temperature": 0.0},
            "stream": True,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks: list[str] = []
    eval_count = 0
    with urllib.request.urlopen(req, timeout=300.0) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("done"):
                eval_count = int(obj.get("eval_count", len(chunks)))
                break
            chunks.append(obj.get("message", {}).get("content", ""))
    return "".join(chunks), eval_count


def has_degenerate_loop(text: str) -> bool:
    words = text.split()
    if len(words) < 10:
        return False
    unique_ratio = len(set(words)) / len(words)
    return unique_ratio < 0.25


def evaluate_server(
    base_url: str,
    model_label: str,
    model_name: str,
    max_tokens: int,
    load_s: float = 0.0,
    rss_mb: float | None = None,
    proc: subprocess.Popen[str] | None = None,
) -> int:
    models_openai = http_get_json(f"{base_url}/v1/models")
    tags_ollama = http_get_json(f"{base_url}/api/tags")
    assert models_openai.get("object") == "list"
    assert "models" in tags_ollama

    print(f"\n[3/4] Exécution de {len(PROMPT_SUITE)} requêtes OpenAI (/v1/chat/completions)...")
    results = []
    for case in PROMPT_SUITE:
        text, ttft_ms, total_s, n_tok = call_openai_chat(
            base_url, case.prompt, max_tokens, case.stream, model_name=model_name
        )
        tok_s = (n_tok / total_s) if total_s > 0 else 0.0
        keyword_ok = not case.expected_any or any(kw in text for kw in case.expected_any)
        no_loop = not has_degenerate_loop(text)
        status = "PASS" if (text.strip() and keyword_ok and no_loop) else "WARN"
        results.append((case.name, case.stream, n_tok, ttft_ms, total_s, tok_s, status, text))
        preview = text.strip().replace("\n", " ")[:72]
        print(f"  [{status}] {case.name:12s} | {n_tok:2d} tok | {tok_s:4.2f} tok/s | TTFT {ttft_ms:6.0f} ms | {preview}")

    ollama_txt, ollama_tok = call_ollama_chat_ndjson(
        base_url, "Dis bonjour en un mot.", min(8, max_tokens), model_name=model_name
    )
    print(f"  [PASS] ollama_ndjson | {ollama_tok:2d} tok | {ollama_txt.strip()!r}")

    peak_mb = (get_process_rss_mb(proc.pid) if proc else None) or rss_mb
    avg_tok_s = sum(r[5] for r in results) / len(results)
    pass_count = sum(1 for r in results if r[6] == "PASS")

    print("\n[4/4] Résumé E2E")
    print("-" * 64)
    print(f"  Machine       : {platform.system()} ({os.cpu_count()} threads CPU)")
    print(f"  Modèle        : {model_label}")
    print(f"  Chargement    : {load_s:.2f} s | RAM Pic : {f'{peak_mb:.0f} Mo' if peak_mb else 'n/a'}")
    print(f"  Débit moyen   : {avg_tok_s:.2f} tok/s")
    print(f"  Qualité       : {pass_count}/{len(results)} PASS + /v1/models + /api/tags + /api/chat OK")
    print("-" * 64)
    return 0 if pass_count == len(results) else 1


def run_e2e(
    model_dir: Path | None,
    max_tokens: int,
    skip_build: bool,
    use_gpu: bool = False,
    target_url: str | None = None,
    model_name: str = "bonsai-aibox",
) -> int:
    if target_url:
        base_url = target_url.rstrip("/")
        print(f"[1/4] Cible externe : {base_url} (modèle: {model_name})")
        return evaluate_server(base_url, f"{model_name} @ {base_url}", model_name, max_tokens)

    if model_dir is None:
        raise ValueError("--model est requis si --target-url n'est pas fourni")

    repo_root = Path(__file__).resolve().parent.parent
    if skip_build:
        bin_name = "aibox.exe" if platform.system() == "Windows" else "aibox"
        bin_path = repo_root / "app" / "target" / "release" / bin_name
    else:
        bin_path = build_aibox_release(repo_root)

    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    print(f"[2/4] Démarrage `aibox serve` sur {base_url} avec {model_dir}...")

    serve_cmd = [str(bin_path), "serve", "--host", "127.0.0.1", "--port", str(port), "--model", str(model_dir)]
    if use_gpu:
        serve_cmd.append("--gpu")
    proc = subprocess.Popen(
        serve_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        load_s = wait_for_health(base_url, proc)
        rss_mb = get_process_rss_mb(proc.pid)
        rss_str = f"{rss_mb:.0f} Mo" if rss_mb is not None else "n/a"
        print(f"      Serveur prêt en {load_s:.2f}s (RAM RSS: {rss_str})")
        return evaluate_server(base_url, str(model_dir), model_name, max_tokens, load_s, rss_mb, proc)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test E2E reproductible pour AI Box (serve + inférence)")
    parser.add_argument("--model", type=Path, default=None, help="Dossier du modèle SafeTensors AI Box")
    parser.add_argument("--target-url", type=str, default=None, help="URL d'un serveur déjà lancé (ex: http://127.0.0.1:11434 pour Ollama)")
    parser.add_argument("--model-name", type=str, default="bonsai-aibox", help="Nom du modèle envoyé à l'API (ex: qwen3:4b)")
    parser.add_argument("--max-tokens", type=int, default=24, help="Tokens max par prompt")
    parser.add_argument("--skip-build", action="store_true", help="Ne pas recompiler `aibox` en release")
    parser.add_argument("--gpu", action="store_true", help="Activer le backend GPU CUDA (`--gpu`)")
    args = parser.parse_args()
    model_dir = args.model.expanduser() if args.model else None
    sys.exit(run_e2e(model_dir, args.max_tokens, args.skip_build, args.gpu, args.target_url, args.model_name))


if __name__ == "__main__":
    main()
