# Установите OpenAI SDK с помощью pip
# pip install openai 
import openai
from openai import OpenAI

YANDEX_CLOUD_FOLDER = ""
YANDEX_CLOUD_API_KEY = ""

def run():
    client = OpenAI(
        api_key=YANDEX_CLOUD_API_KEY,
        base_url="https://llm.api.cloud.yandex.net/v1",
        project=YANDEX_CLOUD_FOLDER
    )

    response = client.chat.completions.create(
        model=f"gpt://{YANDEX_CLOUD_FOLDER}/gpt-oss-120b",
        # или
        # model=f"gpt://{YANDEX_CLOUD_FOLDER}/gpt-oss-20b",
        messages=[
            {
                "role": "developer",
                "content": "Ты очень умный ассистент."},
            {
                "role": "user",
                "content": "Что под капотом LLM?",
            },
        ],
        reasoning_effort="low",
    )

    print(response.choices[0].message.content)

if __name__ == "__main__":
    run()
