from abc import ABC, abstractmethod

from rag_quality_lab.models import RetrievalResult


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]: ...
