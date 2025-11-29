from .inferencer import Inferencer
from tatary.solutions import BaselineSolution


class BaselineAgent(Inferencer):
    async def detoxify(self, texts: list[str]) -> list[str]:
        solution = BaselineSolution()
        responses = await solution.abatch(texts)
        detoxified_texts = [response.detoxified_text for response in responses]
        return detoxified_texts
