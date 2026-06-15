"""Integrations that wire ktune kernels into existing model code."""

from ktune.integrations.patch import apply_ktune_to_model, summarize_patchable

__all__ = ["apply_ktune_to_model", "summarize_patchable"]
