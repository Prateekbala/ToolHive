from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ParameterSpec(BaseModel):
    type: str
    required: bool = False
    description: str = ""
    enum: list[str] | None = None
    default: Any = None


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)

    def required_params(self) -> list[str]:
        return [k for k, v in self.parameters.items() if v.required]

    def to_prompt_dict(self) -> dict[str, Any]:
        """Minimal dict for inclusion in the system prompt."""
        params: dict[str, Any] = {}
        for k, v in self.parameters.items():
            entry: dict[str, Any] = {"type": v.type, "required": v.required}
            if v.description:
                entry["description"] = v.description
            if v.enum:
                entry["enum"] = v.enum
            params[k] = entry
        return {"name": self.name, "description": self.description, "parameters": params}


class DomainSpec(BaseModel):
    domain: str
    version: str = "1.0"
    tools: list[ToolSpec]

    def tool_map(self) -> dict[str, ToolSpec]:
        return {t.name: t for t in self.tools}

    def to_prompt_list(self) -> list[dict[str, Any]]:
        return [t.to_prompt_dict() for t in self.tools]
