# Cloud / free GPU playbook (≤ US$50)

**Decision (2026-11):** no department GPU (#13 closed). Use Colab/Kaggle for smoke; RunPod/Vast only if engine gate needs a real SGLang + 8B bring-up.

## Tier ladder

| Tier | Cost | Use |
|------|------|-----|
| 1. Google Colab (free/T4) | $0 | SGLang install smoke, tiny prompts |
| 2. Kaggle Notebooks (P100/T4 quotas) | $0 | Longer bring-up sessions |
| 3. RunPod / Vast.ai spot | ≤ **$50** hard cap | Equal-citizen gate (#32) if free tiers fail |

## Colab / Kaggle smoke

1. Runtime → GPU (T4 if available).
2. Install: `pip install "sglang[all]"`.
3. Launch server (adjust model to VRAM):
   ```bash
   python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --tp 1
   ```
4. Hit `/v1/chat/completions` with a 2-agent micro-DAG; record tokens/s + whether prefix cache hits appear in metrics.
5. Tear down the runtime when done (Colab disconnects idle sessions).

## Paid cloud (only if needed)

Hard rules:

- Set a **billing alarm at $40**; stop at **$50**.
- Prefer spot/interruptible A10/L4 24GB.
- Log every session in the table below before starting the pod.

### Cost log

| Date | Provider | GPU | $/hr | Hours | Spend | Purpose | Pod ID |
|------|----------|-----|------|-------|-------|---------|--------|
| | | | | | | | |

**Running total:** $0 / $50

### Tear-down checklist

- [ ] Kill SGLang / training processes
- [ ] Stop / terminate pod (not just SSH disconnect)
- [ ] Confirm billing dashboard shows stopped
- [ ] Append row to cost log
- [ ] Commit manifests/metrics under `scheduler/results/` or `engine/results/`

## Relation to Couloir

Sim remains source of truth. GPU track is an optional case study (#29–#32), not a blocker for arXiv draft.
