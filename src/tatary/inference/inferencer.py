from abc import ABC, abstractmethod

class Inferencer(ABC):

    @abstractmethod
    async def detoxify(self, texts: list[str]) -> list[str]:
        return texts
