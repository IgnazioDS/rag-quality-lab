from abc import ABC, abstractmethod

from rag_quality_lab.models import Chunk


class BaseChunker(ABC):
    @property
    @abstractmethod
    def strategy_name(self) -> str: ...

    @abstractmethod
    def chunk(self, text: str, doc_id: str) -> list[Chunk]: ...
