"""Shared Stage 2 artifact naming helpers."""

from __future__ import annotations


DEFAULT_ACTIVATION_SITE = "resid_post"
DEFAULT_HOOK_TEMPLATE = "blocks.{layer}.hook_resid_post"


def normalize_activation_site(site: str | None) -> str:
    if site is None:
        return DEFAULT_ACTIVATION_SITE
    normalized = site.strip().lower().replace("-", "_")
    if not normalized:
        return DEFAULT_ACTIVATION_SITE
    return normalized


def activation_stem(
    *,
    model_key: str,
    task: str,
    layer: int,
    activation_site: str | None = DEFAULT_ACTIVATION_SITE,
) -> str:
    site = normalize_activation_site(activation_site)
    stem = f"{model_key}_{task}_L{layer}"
    if site != DEFAULT_ACTIVATION_SITE:
        stem += f"_{site}"
    return stem


def hook_name_for_layer(*, layer: int, hook_template: str = DEFAULT_HOOK_TEMPLATE) -> str:
    if "{layer}" not in hook_template:
        raise ValueError("--hook-template must contain {layer}")
    return hook_template.format(layer=layer)
