"""
Register the three bundled example domains (inventory, email, crm) in the
specialist registry so ToolHive works immediately after git clone.

Usage:
    python scripts/init_demo.py [--registry registry.db]
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DOMAINS_DIR = REPO_ROOT / "specialists" / "domains"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="registry.db")
    args = parser.parse_args()

    # Registry import after arg parse so path can be set from CLI
    from specialists.registry import SpecialistRegistry, SpecialistEntry

    registry = SpecialistRegistry(db_path=args.registry)
    registry.connect()

    domains = [
        ("inventory", "Qwen/Qwen2.5-3B-Instruct", 0.0),
        ("email",     "Qwen/Qwen2.5-3B-Instruct", 0.0),
        ("crm",       "Qwen/Qwen2.5-3B-Instruct", 0.0),
    ]

    for domain, base_model, score in domains:
        tools_path = DOMAINS_DIR / domain / "tools.yaml"
        if not tools_path.exists():
            print(f"  [skip] {domain}: tools.yaml not found at {tools_path}")
            continue

        existing = registry.list_all()
        already_active = any(
            e.domain == domain and e.status == "active" for e in existing
        )
        if already_active:
            print(f"  [skip] {domain}: already has an active specialist")
            continue

        version = registry.next_version(domain)
        entry = SpecialistEntry(
            specialist_id=f"{domain}-{version}",
            domain=domain,
            base_model=base_model,
            adapter_path="",          # empty → ProviderHarness used at runtime
            tools_yaml_path=str(tools_path),
            eval_score=score,
            trained_at="1970-01-01T00:00:00Z",
            status="candidate",
        )
        registry.register(entry)
        registry.promote(entry.specialist_id)
        print(f"  [ok]   {domain} → {entry.specialist_id} (active)")

    print("Done. Run: python -m pytest tests/ to verify.")


if __name__ == "__main__":
    main()
