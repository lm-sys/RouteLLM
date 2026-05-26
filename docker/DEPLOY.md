# RouteLLM — Docker / K8S Deployment

## Docker Image

### Build

```bash
docker build -f docker/Dockerfile -t routellm:latest .
```

Customize via build args:

```bash
docker build -f docker/Dockerfile \
  --build-arg ROUTERS="sw_ranking causal_llm" \
  --build-arg STRONG_MODEL="gpt-4o" \
  --build-arg PORT=6060 \
  -t routellm:latest .
```

### Run

```bash
docker run -d -p 6060:6060 \
  -e OPENAI_API_KEY="sk-..." \
  -e HF_TOKEN="hf_..." \
  --name routellm routellm:latest
```

With a custom LLM endpoint (e.g. self-hosted proxy):

```bash
docker run -d -p 6060:6060 \
  -e OPENAI_API_KEY="sk-..." \
  -e BASE_URL="http://your-proxy:8080/v1" \
  -e STRONG_MODEL="openai/your-strong-model" \
  -e WEAK_MODEL="openai/your-weak-model" \
  --name routellm routellm:latest
```

With a custom router config file:

```bash
docker run -d -p 6060:6060 \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -e CONFIG_FILE=/app/config.yaml \
  -e OPENAI_API_KEY="sk-..." \
  --name routellm routellm:latest
```

Check: `curl http://localhost:6060/health`  →  `{"status":"online"}`

### Environment Variables

| Variable         | Description                                    | Required                   |
|------------------|------------------------------------------------|----------------------------|
| `OPENAI_API_KEY` | API key for OpenAI-compatible LLMs             | yes                        |
| `HF_TOKEN`       | HuggingFace token (needed when pulling HF models) | no (see note below)     |
| `BASE_URL`       | Base URL for LLM endpoint (default: OpenAI)    | no                         |
| `ROUTERS`        | Space-separated router names                   | no                         |
| `PORT`           | Server port (default 6060)                     | no                         |
| `STRONG_MODEL`   | Strong model ID                                | no                         |
| `WEAK_MODEL`     | Weak model ID                                  | no                         |
| `CONFIG_FILE`    | Path to custom config YAML                     | no                         |

`HF_TOKEN` is only required when routers download models from HuggingFace Hub
at startup. If you use `ROUTERS=random` or connect to an external LLM endpoint
via `BASE_URL`, it is not needed.

---

## K8S Deployment

All manifests are in `docker/deploy/`.

```bash
kubectl apply -f docker/deploy/secret.yaml
kubectl apply -f docker/deploy/configmap.yaml
kubectl apply -f docker/deploy/deployment.yaml
kubectl apply -f docker/deploy/service.yaml
```

Verify:

```bash
kubectl get pods -l app=routellm
kubectl port-forward svc/routellm 6060:6060
curl http://localhost:6060/health
```

### Customizing

- Set `OPENAI_API_KEY` and `HF_TOKEN` in `docker/deploy/secret.yaml`.
- Adjust `STRONG_MODEL` / `WEAK_MODEL` / `ROUTERS` / `BASE_URL` in `docker/deploy/deployment.yaml`.
- If you don't need a custom router config, the ConfigMap is optional
  (the pod will use sensible defaults).

---

## CI (GitHub Actions)

Workflow: `.github/workflows/docker-publish.yml`

- Push to default branch → builds & pushes `:latest` + `:main` + `:<sha>`
- Tag `v*` → pushes `:v1.0.0`, `:1.0`, `:1`
- Pull request → build-only (validates Dockerfile)
- Images published to `ghcr.io/lm-sys/routellm`
