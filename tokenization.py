"""
Four tokenization strategies for multi-turn GRPO.

Strategy           Stateful  Alignment  Notes
-----------------  --------  ---------  ------------------------------------------------
SingleTurn            no        no      Only current question, no history
FullRetokenize        no        no      Rebuild full text from scratch each turn (default)
IncrementalCache      yes       no      Cache prev token IDs, append new user turn only
AlignedCache          yes       yes     IncrementalCache + BPE-boundary alignment

Interface (all strategies):
    tok = make_turn_tokenizer("full")         # or "single" / "incremental" / "aligned"
    tok.reset()                               # call at the start of each episode
    ids = tok.encode_turn(story, qs, as, tokenizer, max_len)   # before each turn
    tok.notify_completion(comp_ids)           # after selecting the best completion
"""

from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Shared prompt builder  (canonical definition — imported by grpo_coqa.py)
# ---------------------------------------------------------------------------

def build_turn_text(
    story: str,
    questions: list[str],
    answers_so_far: list[str],
) -> str:
    text = "Only give short responses:\nContext: " + story
    for q, a in zip(questions[:-1], answers_so_far):
        text += "\nQuestion: " + q + "\nAssistant: " + a.rstrip()
    text += "\nQuestion: " + questions[-1] + "\nAssistant:"
    return text


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class TurnTokenizer(ABC):
    """Common interface for all four tokenization strategies."""

    @abstractmethod
    def encode_turn(
        self,
        story: str,
        questions: list[str],
        answers_so_far: list[str],
        tokenizer,
        max_length: int,
    ) -> list[int]:
        """Return prompt token IDs for this turn (may update internal state)."""

    def notify_completion(self, comp_ids: list[int]) -> None:
        """Inform the strategy of the chosen completion tokens. No-op for stateless strategies."""

    def reset(self) -> None:
        """Reset conversation state. No-op for stateless strategies."""


# ---------------------------------------------------------------------------
# Strategy 1 — SingleTurn
# ---------------------------------------------------------------------------

class SingleTurnTokenizer(TurnTokenizer):
    """
    No history at all.
    Prompt = story + current question only, regardless of turn index.
    """

    def encode_turn(self, story, questions, answers_so_far, tokenizer, max_length) -> list[int]:
        text = "Context: " + story + "\nQuestion: " + questions[-1] + "\nAssistant:"
        return tokenizer.encode(text)[-max_length:]


# ---------------------------------------------------------------------------
# Strategy 2 — FullRetokenize
# ---------------------------------------------------------------------------

class FullRetokenizeTokenizer(TurnTokenizer):
    """
    Rebuild and retokenize the full conversation text from scratch on every turn.
    Consistent tokenization at the cost of O(context_length) work per turn.
    """

    def encode_turn(self, story, questions, answers_so_far, tokenizer, max_length) -> list[int]:
        text = build_turn_text(story, questions, answers_so_far)
        return tokenizer.encode(text)[-max_length:]


# ---------------------------------------------------------------------------
# Strategy 3 — IncrementalCache
# ---------------------------------------------------------------------------

class IncrementalCacheTokenizer(TurnTokenizer):
    """
    Cache the accumulated token IDs across turns.
    Only the new user segment is (re-)tokenized and appended.

    Cost: O(|new_segment|) per turn instead of O(|full_context|).

    Known caveat: BPE splits at the join boundary may differ slightly from
    what FullRetokenize would produce, because the new segment is encoded
    in isolation rather than in full-context.
    """

    def __init__(self):
        self._cache: list[int] = []
        self._eos_id: int | None = None

    def reset(self) -> None:
        self._cache = []

    def notify_completion(self, comp_ids: list[int]) -> None:
        """Append completion tokens to the cache so the next turn sees full history.
        EOS tokens are stripped — they must not appear mid-sequence in the next turn's prompt."""
        eos = self._eos_id
        self._cache.extend(t for t in comp_ids if eos is None or t != eos)

    def encode_turn(self, story, questions, answers_so_far, tokenizer, max_length) -> list[int]:
        if self._eos_id is None:
            self._eos_id = tokenizer.eos_token_id
        if not self._cache:
            # First turn: encode full text with BOS (same as FullRetokenize).
            ids = tokenizer.encode(build_turn_text(story, questions, answers_so_far))
        else:
            # Continuation: encode only the new question segment, no BOS.
            new_ids = tokenizer.encode(
                "\nQuestion: " + questions[-1] + "\nAssistant:",
                add_special_tokens=False,
            )
            ids = self._cache + new_ids

        ids = ids[-max_length:]
        self._cache = list(ids)   # cache stores the prompt; completion appended later via notify_completion
        return ids


# ---------------------------------------------------------------------------
# TokenizationAligner  (helper for Strategy 4)
# ---------------------------------------------------------------------------

class TokenizationAligner:
    """
    Maintains a cumulative split dictionary built from assistant completion tokens.
    Maps surface strings → the token ID sequence the assistant used for them.

    Used to correct BPE boundary inconsistencies when the same surface string
    is encoded in isolation (incremental approach) vs. in full context.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.split_dict: dict[str, list[int]] = {}

    def update(self, assistant_token_ids: list[int]) -> None:
        """
        Scan all 2-to-7-token spans in the assistant output and record their
        surface → token-ID mapping. Most recent occurrence wins on conflict.
        Runtime: O(n) where n = len(assistant_token_ids).
        """
        n = len(assistant_token_ids)
        for start in range(n):
            for end in range(start + 2, min(start + 8, n + 1)):
                span = assistant_token_ids[start:end]
                surface = self.tokenizer.decode(span, skip_special_tokens=True)
                if surface:
                    self.split_dict[surface] = list(span)

    def align(self, text: str) -> list[int]:
        """
        Encode text and replace any windows whose surface form appears in
        split_dict with the canonical token IDs from the assistant.

        Fast path: O(|split_dict|) string check before touching token IDs.
        """
        default_ids = self.tokenizer.encode(text, add_special_tokens=False)

        if not self.split_dict or not any(k in text for k in self.split_dict):
            return default_ids

        max_span = max(len(v) for v in self.split_dict.values())
        aligned, i, n = [], 0, len(default_ids)

        while i < n:
            matched = False
            for span_len in range(min(max_span, n - i), 1, -1):
                window  = default_ids[i : i + span_len]
                surface = self.tokenizer.decode(window, skip_special_tokens=True)
                if surface in self.split_dict and self.split_dict[surface] != window:
                    aligned.extend(self.split_dict[surface])
                    i += span_len
                    matched = True
                    break
            if not matched:
                aligned.append(default_ids[i])
                i += 1

        return aligned

    def reset(self) -> None:
        self.split_dict.clear()


# ---------------------------------------------------------------------------
# Strategy 4 — AlignedCache
# ---------------------------------------------------------------------------

class AlignedCacheTokenizer(IncrementalCacheTokenizer):
    """
    IncrementalCache + aligns new user-turn tokens against the split dictionary
    built from prior assistant completions.

    Corrects BPE boundary inconsistencies introduced by incremental encoding:
    words/subwords that appeared in an assistant turn are guaranteed to receive
    the same token split when they appear in the next user turn.
    """

    def __init__(self, tokenizer):
        super().__init__()
        self._aligner = TokenizationAligner(tokenizer)

    def reset(self) -> None:
        super().reset()
        self._aligner.reset()

    def notify_completion(self, comp_ids: list[int]) -> None:
        self._aligner.update(comp_ids)
        super().notify_completion(comp_ids)

    def encode_turn(self, story, questions, answers_so_far, tokenizer, max_length) -> list[int]:
        if not self._cache:
            # First turn: no prior assistant output to align against.
            return super().encode_turn(story, questions, answers_so_far, tokenizer, max_length)

        new_ids = self._aligner.align("\nQuestion: " + questions[-1] + "\nAssistant:")
        ids = (self._cache + new_ids)[-max_length:]
        self._cache = list(ids)
        return ids


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_turn_tokenizer(strategy: str, tokenizer=None) -> TurnTokenizer:
    """
    Create a TurnTokenizer by name.

    Args:
        strategy:  "single" | "full" | "incremental" | "aligned"
        tokenizer: required only for "aligned"
    """
    if strategy == "single":
        return SingleTurnTokenizer()
    if strategy == "full":
        return FullRetokenizeTokenizer()
    if strategy == "incremental":
        return IncrementalCacheTokenizer()
    if strategy == "aligned":
        if tokenizer is None:
            raise ValueError("AlignedCacheTokenizer requires tokenizer= argument")
        return AlignedCacheTokenizer(tokenizer)
    raise ValueError(f"Unknown tokenization strategy {strategy!r}. Choose: single | full | incremental | aligned")
