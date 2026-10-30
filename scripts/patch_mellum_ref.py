#!/usr/bin/env python3
"""Wire ATOM_REFERENCE_MODE into MellumForCausalLM.forward / compute_logits."""
import ast
from pathlib import Path

p = Path("/workspace/ATOM-tritonmoe/atom/models/mellum.py")
s = p.read_text()
if "ATOM_REFERENCE_MODE" in s:
    print("already wired")
else:
    old = """    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any] | None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model("""
    new = """    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any] | None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if __import__("os").environ.get("ATOM_REFERENCE_MODE") == "1" and inputs_embeds is None:
            from atom.models.reference_mode import reference_forward

            return reference_forward(self, input_ids, positions)
        hidden_states = self.model("""
    assert old in s
    s = s.replace(old, new, 1)

    old2 = """    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        logits = self.lm_head(hidden_states)
        return logits"""
    new2 = """    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if __import__("os").environ.get("ATOM_REFERENCE_MODE") == "1":
            import torch.nn.functional as _F

            return _F.linear(hidden_states, self.lm_head.weight)
        logits = self.lm_head(hidden_states)
        return logits"""
    assert old2 in s
    s = s.replace(old2, new2, 1)
    p.write_text(s)
    ast.parse(s)
    print("mellum.py wired")
