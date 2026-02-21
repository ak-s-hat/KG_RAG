from typing import List, Optional
import torch
from deepeval.models import DeepEvalBaseEmbeddingModel
from sentence_transformers import SentenceTransformer
from config import EVAL_EMBEDDER_MODEL_NAME

class STEmbeddingModel(DeepEvalBaseEmbeddingModel):
    """
    Wraps a sentence-transformers model so DeepEval's Synthesizer
    can use it for document parsing and context grouping.
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        if model_name is None:
            model_name = EVAL_EMBEDDER_MODEL_NAME
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.model_name = model_name
        self.device = device
        print(f"Loading Embedding Model: {self.model_name} on {self.device}")
        self._model = SentenceTransformer(model_name, device=device)

    def load_model(self):
        return self._model

    def get_model_name(self) -> str:
        return self.model_name

    def embed_text(self, text: str) -> List[float]:
        emb = self._model.encode(text, show_progress_bar=False, convert_to_numpy=True)
        return emb.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embs = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [e.tolist() for e in embs]

    async def a_embed_text(self, text: str) -> List[float]:
        return self.embed_text(text)

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embed_texts(texts)