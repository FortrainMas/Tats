from tatary.model import get_model 

from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

class Response(BaseModel):
    detoxified_text: str


# Only model and prompt is used
class BaselineSolution:
    def __init__(self):
        self.model = get_model()
        self.system_prompt = '''You are Gemini, a professional text-detoxification assistant. YOU ARE WORKING ON TEXTS ON TATAR LANGUAGE WHICH YOU KNOW VERY WELL.
Your task is to transform user input into a safe, neutral, respectful, and non-toxic version.
You must:

Remove insults, slurs, aggression, profanity, and harassment.

Preserve the original meaning and intent as much as possible.

If the user requests rewriting, rewrite.

If the user requests analysis, analyze.

If the text is already safe — return it unchanged.

Never add new harmful content.

Always output only the detoxified text, unless the user explicitly asks for explanations.

Your tone must remain neutral and respectful.

Your mission: convert toxic → safe while keeping the message semantically similar and coherent.'''

        self.model = self.model.with_structured_output(Response)

    def invoke(self, text):
        text = f"Here is the text from internet. Please, detoxify it. Please be very close to the original text to keep sence. PLEASE CHECK TWICE THAT IN ANSWER YOU USE TATAR LANGUAGE: {text}."

        request = [
            ("system", self.system_prompt),
            ("human", HumanMessage(content=text))
        ]

        return self.model.invoke(request)




    async def abatch(self, texts: list[str]):

        requests = [
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(
                    content=(
                        "Here is the text from internet. "
                        f"Please, detoxify it. Be close to the original: {text}"
                    )
                )
            ]
            for text in texts
        ]

        return await self.model.abatch(requests)

