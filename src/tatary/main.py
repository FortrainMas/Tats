from tatary.solutions import BaselineSolution
from dotenv import load_dotenv

load_dotenv()

async def main(text: str):
    solution = BaselineSolution()
    result = await solution.abatch([text, text])
    print(result)

BAD_SHIT = '''Син нинди тинтәк кеше!	
Утырасың да күңел ачасың, ахмак!!!!! 
Андый хайваннарны кабер генә төзәтә!
'''




import asyncio


if __name__ == "__main__":
    asyncio.run(main(BAD_SHIT))
