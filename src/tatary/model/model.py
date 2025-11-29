from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel

from tatary.config import get_config

from dotenv import load_dotenv
load_dotenv()

_model = None

def get_model() -> BaseChatModel:
    config = get_config()

    global _model
    if _model is None:
        _model = ChatGoogleGenerativeAI(
            model=config.model.model,
            temperature=config.model.temperature,
            max_tokens=config.model.max_tokens,
        )

    return _model
