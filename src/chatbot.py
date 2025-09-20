# src/chatbot.py
"""
Grounded fashion assistant for StyleFinder AI (chatty + history-aware).

- Defaults to a small chat-tuned model (TinyLlama) for more natural replies.
  You can change model_id to 'distilgpt2' if you prefer the tiny non-chat model.
- Answers are grounded in the retrieved catalog items you pass in.
- Supports short conversation history for a ChatGPT-like feel.

Usage:
    assistant = FashionAssistant()  # or FashionAssistant(model_id="distilgpt2")
    reply = assistant.answer("What goes with black jeans?", retrieved_items_df, history=[...])
"""

from typing import List, Optional, Dict
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


SYSTEM_PROMPT = (
    "You are a friendly, concise fashion assistant. "
    "Ground all advice ONLY in the provided catalog items. "
    "Offer 2–4 specific suggestions and one quick styling tip. "
    "If items are insufficient, ask one clarifying question."
)


def _device_auto() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _format_items(items: pd.DataFrame, max_items: int = 8) -> str:
    """Compact bullet list of retrieved items for grounding."""
    if items is None or items.empty:
        return "- (no items)"
    lines = []
    for _, r in items.head(max_items).iterrows():
        name = (r.get("prod_name") or "").strip()
        ptype = (r.get("product_type_name") or "").strip()
        color = (r.get("colour_group_name") or "").strip()
        aid = r.get("article_id")
        lines.append(f"- {name} ({ptype}, {color}) [id={aid}]")
    return "\n".join(lines)


class FashionAssistant:
    def __init__(
        self,
        model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # more natural, still small
        max_new_tokens: int = 220,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        no_repeat_ngram_size: int = 3,
        device: Optional[str] = None,
    ):
        """
        If you prefer the tiniest download, you can use: model_id='distilgpt2'
        (less natural; not instruction-tuned).
        """
        self.model_id = model_id
        self.device = device or _device_auto()
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.mdl = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size

        # Fallback EOS if model/tok doesn't define it (e.g., GPT2-family)
        self.eos_token_id = (
            self.tok.eos_token_id if self.tok.eos_token_id is not None else 50256
        )

    def _build_messages(
        self, user_question: str, retrieved_items: pd.DataFrame, history: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """Create chat-style messages list (system + short history + current turn)."""
        msgs: List[Dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            # keep it short to avoid long prompts
            for m in history[-6:]:
                role = "assistant" if m.get("role") == "assistant" else "user"
                content = m.get("content") or m.get("text") or ""
                if content:
                    msgs.append({"role": role, "content": content})

        catalog = _format_items(retrieved_items)
        msgs += [
            {"role": "user", "content": f"My question: {user_question}"},
            {"role": "user", "content": f"Relevant catalog items:\n{catalog}"},
            {"role": "user", "content": "Please answer using ONLY the items above."},
        ]
        return msgs

    @torch.no_grad()
    def answer(
        self,
        user_question: str,
        retrieved_items: pd.DataFrame,
        history: Optional[List[Dict]] = None,
    ) -> str:
        """
        Generate a short, grounded answer. Uses chat template if the tokenizer provides one;
        otherwise falls back to a simple instruction prompt.
        """
        # Try to use chat template (supported by chat-tuned models like TinyLlama).
        msgs = self._build_messages(user_question, retrieved_items, history)

        try:
            prompt = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tok(prompt, return_tensors="pt").to(self.device)
            input_len = inputs.input_ids.shape[1]

            output_ids = self.mdl.generate(
                **inputs,
                do_sample=True,
                top_p=self.top_p,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                repetition_penalty=self.repetition_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                eos_token_id=self.eos_token_id,
            )
            # Take only the newly generated tokens (after the prompt)
            gen_ids = output_ids[0, input_len:]
            text = self.tok.decode(gen_ids, skip_special_tokens=True).strip()

        except Exception:
            # Fallback: build a plain prompt (works for distilgpt2)
            catalog = _format_items(retrieved_items)
            plain_prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"User question:\n{user_question}\n\n"
                f"Relevant catalog items:\n{catalog}\n\n"
                "Assistant (use only these items):"
            )
            inputs = self.tok(plain_prompt, return_tensors="pt").to(self.device)
            output_ids = self.mdl.generate(
                **inputs,
                do_sample=True,
                top_p=self.top_p,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                repetition_penalty=self.repetition_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                eos_token_id=self.eos_token_id,
            )
            text_full = self.tok.decode(output_ids[0], skip_special_tokens=True)
            # return only after the marker
            marker = "Assistant (use only these items):"
            text = text_full.split(marker, 1)[-1].strip() if marker in text_full else text_full.strip()

        # Light cleanup & brevity
        text = " ".join(text.split())             # collapse long whitespace
        if len(text) > 900:                       # keep answers tight for UI
            text = text[:900].rsplit(". ", 1)[0] + "."

        return text or "I need a bit more detail about the occasion or style to refine suggestions."
