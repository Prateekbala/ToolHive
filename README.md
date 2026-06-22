# ToolHive

**A self-healing swarm of specialist tool-calling agents.**

One base model. Many domain-specific LoRA adapters paged in and out by vLLM's S-LoRA layer. Every tool call is verified by a 3-layer critic. Failures automatically feed a retrain flywheel that patches each specialist overnight — no human labeling, no manual redeployment.

---

## Why ToolHive

Most LLM tool-calling setups hit the same ceiling: a single general-purpose model can't be an expert at every API domain simultaneously. It hallucinates parameters, picks the wrong tool, or produces malformed JSON — and fixing it requires someone to notice, label data, retrain, and redeploy. Manually. Every time.

ToolHive solves this three ways simultaneously:

- **Specialists, not generalists.** Each domain (inventory, email, CRM, billing…) gets its own fine-tuned LoRA adapter, purpose-built to call that domain's tools accurately.
- **Shared GPU, zero waste.** All adapters share one base model in GPU memory via vLLM S-LoRA. Swapping specialists takes microseconds, not seconds — you can run dozens of specialists on a single GPU that would otherwise need one each.
- **Self-healing.** The critic catches failures in real time. The flywheel clusters them nightly, generates targeted fix examples, retrains the adapter, and promotes it automatically — only if it scores higher than the previous version.

---

## What you can build with it

**Enterprise API orchestration** — Route "check PO status then update the CRM" across ERP, CRM, and email systems. Each domain gets its own specialist; the router splits multi-step requests and dispatches sub-queries independently.

**Internal developer tooling** — Build a natural-language interface over your internal APIs. Add a new domain in minutes with a `tools.yaml` file; the flywheel handles long-tail edge cases as they accumulate in production.

**Domain-specific AI assistants** — Deploy a specialist that only knows inventory management. Narrower scope means higher accuracy and fewer hallucinations than a general model trying to cover everything.

**Multi-system automation** — Chain actions across Jira, Slack, GitHub, and Salesforce without hard-coding integration logic for every permutation. The router figures out which specialist handles which sub-request.

**Self-improving agents** — Use ToolHive as the tool-dispatch layer for a larger agentic system. As your API surfaces change, specialists adapt automatically rather than degrading silently.

---

## How it works

```
User request
      │
      ▼
   Router  ──── embedding-similarity dispatch ──►  Specialists
   (splits multi-step requests automatically)       one LoRA adapter per domain
                                                    all sharing one GPU (S-LoRA)
                                                          │
                                                    Critic verifier
                                                    1. Schema check   (pure Python)
                                                    2. Semantic check (LLM)
                                                    3. Escalation     (larger model)
                                                          │
                                              ┌───────────┴────────────────┐
                                              ▼                            ▼
                                         Response               Feedback store
                                                                      │ failures
                                                                      ▼
                                                            Retrain flywheel
                                                            cluster → patch
                                                            → eval → promote
```

**Router** — Embeds the request against each specialist's tool-description corpus. When it detects connectors ("check stock *then* email the supplier"), it splits the request into sub-queries and dispatches each to the right domain independently.

**Specialists** — Fine-tuned LoRA adapters on top of a shared base model (default: `Qwen/Qwen2.5-3B-Instruct`). Each adapter is trained on synthetic examples generated specifically for its domain, then patched continuously from production failures.

**Critic** — Three-layer verifier that runs after every specialist response. Layer 1 is pure Python schema validation (required params, types, enum values). Layer 2 sends a scrubbed version of the tool call to an LLM for semantic plausibility. Layer 3 escalates ambiguous verdicts to a larger model. Prompts are pinned constants — never assembled dynamically — to prevent verdict drift from phrasing changes.

**Feedback store** — Append-only SQLite log of every request, tool call, and critic verdict. The retrain flywheel reads from this. No manual labeling required; critic flags substitute for labels.

**Retrain flywheel** — Runs nightly per specialist. Clusters failures with HDBSCAN, generates targeted patch examples via the provider API, filters them with a virtual judge, retrains the adapter, evaluates on the *full* held-out set (not just recent failures — to catch regressions), and promotes only if the new adapter scores ≥ the previous one. If live accuracy drops post-promotion, the registry rolls back automatically.

---

## Quickstart — no GPU required

When `TOOLHIVE_PROVIDER_API_KEY` is set, specialists use the provider API for inference instead of loading local adapters. Works with OpenAI, Groq, Together, Ollama, or any OpenAI-compatible endpoint.

```bash
git clone https://github.com/your-org/toolhive
cd toolhive
pip install -e ".[pipeline,router]" && pip install fastapi uvicorn httpx

cp .env.example .env
# Add TOOLHIVE_PROVIDER_API_KEY to .env

python scripts/init_demo.py      # registers the bundled inventory, email, crm domains

uvicorn server.app:app --port 8000 &
uvicorn dashboard.api:app --port 8080
```

Or with Docker:

```bash
cp .env.example .env && docker-compose up
```

**Send a request:**

```bash
curl -X POST http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -d '{"request": "how many units of SKU-123 are left in warehouse WH-A?"}'
```

```json
{
  "success": true,
  "results": [{
    "specialist_id": "inventory-v1",
    "tool_name": "get_inventory",
    "parameters": { "product_id": "SKU-123", "warehouse": "WH-A" },
    "critic_verdict": "pass"
  }]
}
```

Multi-step requests are split automatically:

```bash
curl -X POST http://localhost:8000/invoke \
  -d '{"request": "check stock for SKU-123 then email the supplier"}'
# → two results: inventory specialist + email specialist, dispatched independently
```

Open **http://localhost:8080** for the live dashboard — per-specialist accuracy over time, failure groups by critic reason, and retrain history.

---

## Adding a domain

Create two files:

```yaml
# specialists/domains/billing/tools.yaml
domain: billing
version: "1.0"
tools:
  - name: get_invoice
    description: Retrieve an invoice by ID.
    parameters:
      invoice_id: { type: string, required: true }
  - name: no_tool
    description: Use when no billing tool fits.
    parameters: {}
```

```markdown
# specialists/domains/billing/goal.md
## Performance targets
- tool_selection_accuracy: ≥ 95%
- hallucination_rate: < 2%
- adapter_rank: 16
```

Register and activate:

```python
from specialists.registry import SpecialistRegistry, SpecialistEntry
r = SpecialistRegistry("registry.db")
r.connect()
e = SpecialistEntry(
    specialist_id="billing-v1", domain="billing",
    base_model="Qwen/Qwen2.5-3B-Instruct",
    adapter_path="",   # empty = provider API until fine-tuned
    tools_yaml_path="specialists/domains/billing/tools.yaml",
    eval_score=0.0, trained_at="1970-01-01T00:00:00Z",
)
r.register(e)
r.promote("billing-v1")
```

```bash
curl -X POST http://localhost:8000/reload   # live, no restart needed
```

---

## Fine-tuning (GPU)

```bash
# Generate 500 synthetic training examples for a domain
python -m pipeline.loop --domain specialists/domains/inventory --examples 500

# Trigger a retrain cycle
python -m scheduler.retrain \
  --specialist inventory-v1 \
  --registry registry.db --feedback feedback.db --state scheduler_state.db
```

The scheduler auto-triggers when ≥ 20 critic failures accumulate since the last run.

---

## Tests

```
333 passed in ~1 second
```

| Area | Tests |
|---|---|
| Inference harness (parse, validate, schema) | 25 |
| Training pipeline (datagen, cluster, eval, providers) | 62 |
| Specialist registry (CRUD, promote, rollback) | 14 |
| Router (embedding, multi-step, orchestrator) | 47 |
| Critic verifier (schema, semantic, escalation, PII, calibration) | 48 |
| Retrain flywheel (scheduler, live monitor, state store) | 38 |
| Dashboard & alerting (metrics, API, webhooks) | 50 |
| Packaging & integration (domains, server, docker-compose) | 49 |

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q
```

---

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `/invoke` | POST | Route and execute a request. Returns tool call + critic verdict per step. |
| `/domains` | GET | List active specialists and their eval scores. |
| `/reload` | POST | Rebuild the router index after registering new specialists. |
| `/health` | GET | Health check with active specialist IDs. |
| `dashboard:8080/` | GET | Observability dashboard. |
| `dashboard:8080/specialists/{id}/accuracy` | GET | Accuracy over time (24 hourly buckets). |
| `dashboard:8080/specialists/{id}/failures` | GET | Failure groups by critic reason. |
| `dashboard:8080/router/accuracy` | GET | Overall and per-specialist routing accuracy. |

Full interactive docs at `http://localhost:8000/docs`.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `TOOLHIVE_PROVIDER_API_KEY` | — | API key. If set, uses the provider for inference (no GPU needed). |
| `TOOLHIVE_PROVIDER_MODEL` | `gpt-4o-mini` | Model name sent to the provider. |
| `TOOLHIVE_PROVIDER_BASE_URL` | `https://api.openai.com/v1` | Provider base URL. |
| `TOOLHIVE_BASE_MODEL` | `Qwen/Qwen2.5-3B-Instruct` | Base model for local GPU LoRA inference. |
| `TOOLHIVE_CRITIC_ENABLED` | `1` | Set to `0` to disable the critic. |
| `TOOLHIVE_ALERT_WEBHOOK` | — | Webhook URL for failure-rate alerts (Slack, PagerDuty, etc.). |
| `TOOLHIVE_ALERT_THRESHOLD` | `0.20` | Failure rate that fires the alert. |
| `TOOLHIVE_REGISTRY_DB` | `registry.db` | Specialist registry (SQLite). |
| `TOOLHIVE_FEEDBACK_DB` | `feedback.db` | Feedback store (SQLite). |
| `TOOLHIVE_STATE_DB` | `scheduler_state.db` | Scheduler state (SQLite). |

---

## Design notes

**All adapters use LoRA rank 16.** Co-serving adapters of different ranks forces vLLM's fused kernels to pad compute tiles to the maximum rank present, degrading lower-rank adapters' P95 latency by up to 84% (LoRAServe, arXiv:2511.22880).

**Critic prompts are pinned constants.** Prompt-template sensitivity causes LLM judge verdict flip rates of 0.4–0.99 with minor phrasing changes (JudgeSense, 2025). Dynamic assembly of critic prompts would corrupt the feedback labels and destabilize the flywheel. Precision must be ≥ 85% on a labeled gold set before the critic's verdicts are allowed to trigger retraining.

**No production data leaves the host.** External provider calls during retraining use only synthetic prompts derived from failure cluster descriptions. PII (email, phone, SSN, IP) is regex-scrubbed before any content crosses the network boundary.



