import os
from typing import Literal, Union

from pydantic import BaseModel, Field, ValidationError


from dotenv import load_dotenv

load_dotenv()

class GeminiConfig(BaseModel):
    type: Literal["Gemini"]

    model: str
    temperature: float
    max_tokens: int
    API_KEY: str

class Config(BaseModel):
    model: GeminiConfig



google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key is None:
    raise Exception("GOOGLE_API_KEY is not set")

config = {
    "model": {
        "type": "Gemini",
        "model": "gemini-2.5-flash",
        "temperature": 0.7,
        "max_tokens": 100000,
        "API_KEY": google_api_key
    }
}


_config = None
def get_config() -> Config:
    global _config
    global config
    if _config is not None:
        return _config


    try:
        _config = Config(**config)
    except ValidationError as e:
        raise Exception(f"Invalid config: {e}")

    return _config
