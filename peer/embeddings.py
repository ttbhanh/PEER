from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import joblib

from .utils import ensure_dir


class TextEmbedder:
    """Sentence-transformers wrapper with a deterministic TF-IDF/SVD fallback."""

    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2', device: str = 'cuda', fp16: bool = True, fallback_tfidf: bool = True):
        self.model_name = model_name
        self.device = device
        self.fp16 = fp16
        self.fallback_tfidf = fallback_tfidf
        self.kind = 'sentence_transformer'
        self.model = None
        self.vectorizer = None
        if str(model_name).lower().startswith(('tfidf', 'none', 'fallback')):
            self.kind = 'tfidf_svd'
            return
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=device)
            if fp16:
                try:
                    self.model.half()
                except Exception:
                    pass
        except Exception as e:
            if not fallback_tfidf:
                raise
            self.kind = 'tfidf_svd'
            self.model = None
            self._load_error = repr(e)

    def fit(self, texts: list[str], n_components: int = 384):
        if self.kind == 'tfidf_svd':
            max_features = min(50000, max(1000, len(texts) * 10))
            n_components = min(n_components, max(2, len(texts) - 1), max_features - 1)
            self.vectorizer = make_pipeline(
                TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=1),
                TruncatedSVD(n_components=n_components, random_state=42),
                Normalizer(copy=False),
            )
            self.vectorizer.fit(texts)
        return self

    def encode(self, texts: list[str], batch_size: int = 512) -> np.ndarray:
        if self.kind == 'sentence_transformer':
            n = len(texts)
            dim = self.model.get_sentence_embedding_dimension()
            out = np.empty((n, dim), dtype=np.float32)
            # Encode in large chunks (not one call for the whole corpus): keeps peak
            # memory bounded to one chunk instead of accumulating a Python list of
            # per-batch arrays and concatenating at the end, and prints progress we
            # explicitly flush -- the library's own progress bar can appear to hang
            # when stdout is redirected to a file for a long (tens of minutes) run.
            chunk_size = max(batch_size * 4, 2000)
            for start in range(0, n, chunk_size):
                end = min(n, start + chunk_size)
                arr = self.model.encode(
                    texts[start:end],
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                out[start:end] = arr.astype(np.float32)
                print(f'Encoded {end}/{n} texts', flush=True)
            return out
        if self.vectorizer is None:
            self.fit(texts)
        return self.vectorizer.transform(texts).astype(np.float32)

    def save(self, path: str | Path):
        path = Path(path)
        ensure_dir(path)
        meta = {'kind': self.kind, 'model_name': self.model_name}
        (path / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        if self.kind == 'tfidf_svd':
            joblib.dump(self.vectorizer, path / 'tfidf_svd.joblib')


def _npy_and_ids_paths(path: str | Path) -> tuple[Path, Path]:
    """embeddings.npz -> (embeddings.npy, embeddings.ids.json). A plain .npy
    array file (not zip-wrapped like .npz, even uncompressed) is the only
    format numpy can actually mmap -- np.load(path, mmap_mode='r') on a .npz
    member always fully materializes it in RAM regardless of mmap_mode,
    since a zip entry isn't a directly-mappable OS file. Splitting the (small)
    ids list into its own JSON file keeps the big array file mmap-clean."""
    path = Path(path)
    return path.with_suffix('.npy'), path.with_suffix('.ids.json')


def save_embeddings(ids: list[str], embeddings: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    npy_path, ids_path = _npy_and_ids_paths(path)
    np.save(npy_path, embeddings.astype(np.float32, copy=False))
    with open(ids_path, 'w') as f:
        json.dump([str(i) for i in ids], f)


def load_embeddings(path: str | Path) -> dict[str, np.ndarray]:
    npy_path, ids_path = _npy_and_ids_paths(path)
    emb = np.load(npy_path, mmap_mode='r')
    with open(ids_path) as f:
        ids = json.load(f)
    return {str(i): emb[idx] for idx, i in enumerate(ids)}
