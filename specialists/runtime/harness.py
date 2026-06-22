"""
Phase 0: Inference harness — load base model + LoRA adapter, run a tool call.

CLI usage:
  python -m specialists.runtime.harness \\
    --model Qwen/Qwen2.5-3B-Instruct \\
    --tools specialists/domains/inventory/tools.yaml \\
    --query "How many units of SKU-123 are left in warehouse WH-A?" \\
    [--adapter path/to/adapter] \\
    [--empty-adapter]          # Phase 0 smoke test: attach untrained adapter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .parser import parse_tool_call, validate_against_spec
from .schema import DomainSpec

# Pinned system prompt — never dynamically assembled (critic bias mitigation).
_SYSTEM_PROMPT = """\
You are a precise tool-calling assistant. Given a user request, output ONLY a \
single JSON object representing the best tool call. No explanation, no markdown — \
just the raw JSON on one line.

Available tools:
{tools_json}

Output format:
{{"name": "<tool_name>", "parameters": {{<param>: <value>, ...}}}}

Rules:
- Only include parameters explicitly mentioned or clearly implied by the request.
- Never invent or guess parameter values.
- Omit optional parameters that are not mentioned.
- If no tool matches, output: {{"name": "no_tool", "parameters": {{}}}}
"""


def load_domain(tools_yaml_path: str | Path) -> DomainSpec:
    path = Path(tools_yaml_path)
    with path.open() as f:
        data = yaml.safe_load(f)
    return DomainSpec.model_validate(data)


class ToolCallHarness:
    """
    Wraps a base LLM (+ optional LoRA adapter) for structured tool-call inference.

    All adapters must use the same LoRA rank (see goal.md) so the S-LoRA
    Unified Paging serving layer can co-serve them without rank-heterogeneity
    tail-latency penalties.
    """

    def __init__(
        self,
        model_name_or_path: str,
        adapter_path: str | None = None,
        device: str = "auto",
    ):
        self.model_name_or_path = model_name_or_path
        self.adapter_path = adapter_path
        self.device = device
        self.model: Any = None
        self.tokenizer: Any = None

    def load(self) -> None:
        """Load base model + tokenizer, then attach LoRA adapter if provided."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[harness] loading base model: {self.model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, trust_remote_code=True
        )
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        if self.adapter_path:
            from peft import PeftModel

            print(f"[harness] attaching adapter: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

        self.model.eval()
        print("[harness] ready")

    def init_empty_adapter(self, lora_rank: int = 16) -> None:
        """
        Attach an untrained LoRA adapter for Phase 0 end-to-end smoke testing.
        Output will be inaccurate (random delta) but the full pipeline runs.
        """
        from peft import LoraConfig, TaskType, get_peft_model

        if self.model is None:
            raise RuntimeError("call load() before init_empty_adapter()")

        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        self.model = get_peft_model(self.model, config)
        self.model.eval()
        print(f"[harness] empty LoRA adapter initialized (rank={lora_rank})")

    def run(
        self,
        domain: DomainSpec,
        query: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """
        Format prompt → generate → parse → validate.

        Returns:
          name          : tool name (or None if parse failed)
          parameters    : parameter dict
          raw           : raw model output string
          parse_error   : True if JSON could not be extracted
          schema_errors : list of schema violations (empty = valid)
        """
        import torch

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("call load() first")

        tools_json = json.dumps(domain.to_prompt_list(), indent=2)
        system_content = _SYSTEM_PROMPT.format(tools_json=tools_json)

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        parsed = parse_tool_call(raw)
        schema_errors: list[str] = []
        if parsed:
            schema_errors = validate_against_spec(parsed, domain.tool_map())

        return {
            "name": parsed["name"] if parsed else None,
            "parameters": parsed.get("parameters", {}) if parsed else {},
            "raw": raw,
            "parse_error": parsed is None,
            "schema_errors": schema_errors,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="ToolHive Phase 0 inference harness")
    parser.add_argument("--model", required=True, help="HuggingFace model name or local path")
    parser.add_argument("--tools", required=True, help="Path to domain tools.yaml")
    parser.add_argument("--query", required=True, help="User query to route to a tool")
    parser.add_argument("--adapter", default=None, help="Path to trained LoRA adapter directory")
    parser.add_argument(
        "--empty-adapter",
        action="store_true",
        help="Attach an untrained LoRA adapter (Phase 0 smoke test)",
    )
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank for empty adapter")
    parser.add_argument("--device", default="auto", help="torch device_map (auto/cpu/cuda)")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    domain = load_domain(args.tools)
    harness = ToolCallHarness(args.model, adapter_path=args.adapter, device=args.device)
    harness.load()

    if args.empty_adapter:
        harness.init_empty_adapter(lora_rank=args.rank)

    result = harness.run(domain, args.query, max_new_tokens=args.max_tokens)
    print(json.dumps(result, indent=2))

    if result["parse_error"]:
        raise SystemExit(1)
    if result["schema_errors"]:
        print(f"[warn] schema errors: {result['schema_errors']}", flush=True)


if __name__ == "__main__":
    main()
