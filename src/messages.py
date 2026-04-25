"""Prompt message helpers shared by inference and Stage 2 extraction."""

from __future__ import annotations


def build_messages(system: str, user: str, model_name: str) -> list[dict[str, str]]:
    """Construct chat messages, handling Gemma's lack of a system role.

    Gemma 2/3 tokenizers do not define a ``system`` role in the chat template.
    Stage 1 served Gemma through vLLM by concatenating the system and user
    prompt into one user message; Stage 2 must reconstruct that same prompt
    before extracting pre-CoT activations.
    """
    if "gemma" in model_name.lower():
        combined = f"{system}\n\n{user}"
        return [{"role": "user", "content": combined}]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def render_chat_text(
    tokenizer,
    *,
    system: str,
    user: str,
    model_name: str,
    add_generation_prompt: bool = True,
) -> str:
    """Render the exact chat-template text used for activation extraction."""
    messages = build_messages(system, user, model_name)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
