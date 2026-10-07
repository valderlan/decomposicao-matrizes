from dataclasses import dataclass, field

import numpy as np


class MethodError(ValueError):
    """O algoritmo escolhido não pode prosseguir sob as hipóteses informadas."""


@dataclass
class Step:
    title: str
    explanation: str
    formula: str = ""
    matrices: dict[str, np.ndarray] = field(default_factory=dict)
    calculations: list[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "title": self.title,
            "explanation": self.explanation,
            "formula": self.formula,
            "calculations": self.calculations,
            "matrices": {name: value.tolist() for name, value in self.matrices.items()},
        }


@dataclass
class Trace:
    enabled: bool = True
    steps: list[Step] = field(default_factory=list)

    def add(self, title, explanation, formula="", *, calculations=None, **matrices):
        if self.enabled:
            self.steps.append(
                Step(
                    title,
                    explanation,
                    formula,
                    {k: np.array(v, copy=True) for k, v in matrices.items()},
                    list(calculations or []),
                )
            )


@dataclass
class Decomposition:
    method: str
    identity: str
    factors: dict[str, np.ndarray]
    steps: list[Step]
    relative_error: float
    warnings: list[str] = field(default_factory=list)
    orthogonality_error: float | None = None

    def to_dict(self):
        return {
            "method": self.method,
            "identity": self.identity,
            "factors": {k: v.tolist() for k, v in self.factors.items()},
            "steps": [s.to_dict() for s in self.steps],
            "relative_error": self.relative_error,
            "orthogonality_error": self.orthogonality_error,
            "warnings": self.warnings,
        }


@dataclass
class Solution:
    x: np.ndarray
    residual_norm: float
    relative_residual: float
    kind: str
    steps: list[Step]

    def to_dict(self):
        return {
            "x": self.x.tolist(),
            "residual_norm": self.residual_norm,
            "relative_residual": self.relative_residual,
            "kind": self.kind,
            "steps": [s.to_dict() for s in self.steps],
        }
