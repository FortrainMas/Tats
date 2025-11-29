import pandas as pd
import asyncio
from tatary.inference import BaselineAgent

from os import getcwd

DATA_PATH = getcwd() + "/data/dev_inputs.tsv"
ANS_PATH = getcwd() + "/data/dev_outputs.tsv"



import time

async def main():
    print("ID KILL MYSELF")
    df = pd.read_csv(DATA_PATH, sep="\t")

    texts = df["tat_toxic"].tolist()
    texts_detoxified = []

    step = 10
    for i in range(0, len(texts)+step-1, step):
        current_texts = texts[i : min(len(texts),i + step)]
        print(current_texts)
        start = time.perf_counter()
        try:
            texts_detoxified.extend(await BaselineAgent().detoxify(current_texts))
        except Exception as e:
            exit()

        df["tat_detox1"] = texts_detoxified
        df.to_csv(ANS_PATH, sep="\t", index=False)
        
        await asyncio.sleep(120)
        end = time.perf_counter()
        print(f"Batch {i} took {end - start:.3f} seconds")
        print(i)


    df["tat_detox1"] = texts_detoxified

    df.to_csv(ANS_PATH, sep="\t", index=False)


if __name__ == "__main__":
    asyncio.run(main())
