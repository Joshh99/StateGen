# agents/memory_manager.py
# Stores successful (state, context) → code mappings.
# Retrieves them by semantic similarity so similar future
# states can reuse what already worked.

import json
import logging
from typing import List, Optional
import numpy as np
from pathlib import Path

from config import EMBEDDING_MODEL, SIM_THRESHOLD, MEMORY_TOP_K, MEMORY_FILE
from core.models import State, ProgramContext, CodePattern

logger = logging.getLogger(__name__)

# Libraries whose names are injected as tags into the embedding query so that
# a Pandas solution is never retrieved for a PyTorch state (and vice-versa).
_KNOWN_LIBRARIES = [
    "pandas", "numpy", "scipy", "sklearn", "scikit-learn",
    "torch", "pytorch", "tensorflow", "keras",
    "matplotlib", "seaborn", "plotly",
    "statsmodels", "xarray",
    "PIL", "pillow", "cv2", "opencv",
    "requests", "flask", "fastapi", "sqlalchemy",
    "nltk", "spacy", "transformers", "huggingface",
    "boto3", "s3", "gcp", "azure",
]


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
        Return the single best-matching code pattern, or None.
        Use lookup_top_k() to get multiple patterns for few-shot injection.
        """
        results = self.lookup_top_k(state, context, k=1)
        return results[0] if results else None

    def lookup_top_k(
        self, state: State, context: ProgramContext, k: int = MEMORY_TOP_K
    ) -> List[str]:
        """
        Return up to k verified code patterns whose embedding similarity exceeds
        the threshold, sorted by similarity (best first).

        Library tags are embedded in the query, so a Pandas pattern will not be
        retrieved for a PyTorch state even if their descriptions are otherwise similar.
        """
        if not self.patterns:
            return []

        query     = self._query_text(state, context)
        query_emb = self._embed(query)

        scored: List[tuple] = []
        for pattern in self.patterns:
            if pattern.embedding is None:
                continue
            sim = self._cosine(query_emb, pattern.embedding)
            if sim >= self.threshold:
                scored.append((sim, pattern.code))

        scored.sort(key=lambda x: -x[0])
        top = [code for _, code in scored[:k]]

        if top:
            logger.info(
                f"Memory top-{k}: {len(top)} hit(s) "
                f"(best={scored[0][0]:.2f}) for {state}"
            )
        else:
            logger.debug(
                f"No memory hit for {state} "
                f"(best={scored[0][0]:.2f})" if scored else f"(no patterns)"
            )
        return top

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
        """
        Build the embedding query string.

        Library tags (e.g. [pandas] [numpy]) are prepended so that cosine
        similarity is penalised across library boundaries — a Pandas import
        pattern won't be retrieved for a PyTorch state.
        """
        combined = f"{state.description} {self._context_sig(context)}"
        tags     = self._detect_libraries(combined)
        tag_str  = " ".join(f"[{t}]" for t in tags)
        return f"{tag_str} {state.description} | {self._context_sig(context)}".strip()

    @staticmethod
    def _detect_libraries(text: str) -> List[str]:
        """Return known library names found in text (lowercased comparison)."""
        text_lower = text.lower()
        return [lib for lib in _KNOWN_LIBRARIES if lib in text_lower]

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
