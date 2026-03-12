# agents/memory_manager.py
# Stores successful (state, context) → code mappings.
# Retrieves them by semantic similarity so similar future
# states can reuse what already worked.

import json
import logging
from typing import List, Optional
import numpy as np
from pathlib import Path

from config import EMBEDDING_MODEL, SIM_THRESHOLD, MEMORY_FILE
from core.models import State, ProgramContext, CodePattern

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Stores and retrieves successful code patterns.

    Usage:
        memory = MemoryManager()

        # Look up before generating
        cached = memory.lookup(state, context)
        if cached:
            use cached
        else:
            generate, verify, then...
            memory.store(state, context, code)
    """

    def __init__(self, threshold: float = SIM_THRESHOLD):
        self.threshold = threshold
        self.patterns: List[CodePattern] = []
        self._embedder = None       # lazy-loaded
        self._load_from_disk()

    # ─────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────

    def lookup(self, state: State, context: ProgramContext) -> Optional[str]:
        """
        Search for a matching code pattern.

        Returns the cached code string if a match is found,
        otherwise returns None.
        """
        if not self.patterns:
            return None

        query        = self._query_text(state, context)
        query_emb    = self._embed(query)
        best_code    = None
        best_sim     = 0.0

        for pattern in self.patterns:
            if pattern.embedding is None:
                continue
            sim = self._cosine(query_emb, pattern.embedding)
            if sim > best_sim:
                best_sim  = sim
                best_code = pattern.code

        if best_sim >= self.threshold:
            logger.info(f"Memory hit (similarity={best_sim:.2f}) for {state}")
            return best_code

        logger.debug(f"No memory hit (best={best_sim:.2f}) for {state}")
        return None

    def store(self, state: State, context: ProgramContext, code: str):
        """
        Save a successful code pattern.
        Updates the count if the pattern already exists.
        """
        desc = state.description
        sig  = self._context_sig(context)

        # Update existing pattern if it matches exactly
        for pattern in self.patterns:
            if pattern.state_description == desc and pattern.context_signature == sig:
                pattern.success_count += 1
                logger.debug(f"Updated pattern count → {pattern.success_count}")
                self._save_to_disk()
                return

        # New pattern
        query = self._query_text(state, context)
        emb   = self._embed(query)

        self.patterns.append(
            CodePattern(
                state_description=desc,
                context_signature=sig,
                code=code,
                success_count=1,
                embedding=emb,
            )
        )
        logger.info(f"Stored new pattern (total={len(self.patterns)})")
        self._save_to_disk()

    def stats(self) -> dict:
        return {
            "total_patterns": len(self.patterns),
            "total_reuses": sum(p.success_count - 1 for p in self.patterns),
        }

    # ─────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────

    def _query_text(self, state: State, context: ProgramContext) -> str:
        """Combine state description + context signature into one string."""
        return f"{state.description} | {self._context_sig(context)}"

    @staticmethod
    def _context_sig(context: ProgramContext) -> str:
        """
        Deterministic signature of what variables are in scope.
        e.g. "df:DataFrame, n:number"
        """
        return ", ".join(
            f"{k}:{v}" for k, v in sorted(context.variables.items())
        )

    def _embed(self, text: str) -> np.ndarray:
        """Compute sentence embedding (lazy-loads the model on first call)."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedder: {EMBEDDING_MODEL}")
            self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedder.encode(text, convert_to_numpy=True)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    # ─────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────

    def _save_to_disk(self):
        path = Path(MEMORY_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [
            {
                "state_description": p.state_description,
                "context_signature": p.context_signature,
                "code": p.code,
                "success_count": p.success_count,
                # embeddings are recomputed on load — no need to store
            }
            for p in self.patterns
        ]

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_from_disk(self):
        path = Path(MEMORY_FILE)
        if not path.exists():
            return

        with open(path) as f:
            data = json.load(f)

        for item in data:
            self.patterns.append(
                CodePattern(
                    state_description=item["state_description"],
                    context_signature=item["context_signature"],
                    code=item["code"],
                    success_count=item["success_count"],
                    embedding=None,   # recomputed lazily on first lookup
                )
            )
        logger.info(f"Loaded {len(self.patterns)} patterns from disk")
