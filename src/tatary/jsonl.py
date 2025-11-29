import pandas as pd
import json

from os import getcwd

DATA_PATH = getcwd() + "/data/dev_inputs.tsv"
JSONL_PATH = getcwd() + "/data/tat_texts.jsonl"

df = pd.read_csv(DATA_PATH, sep="\t")


instruction = """
You are an expert assistant in Tatar language text detoxification. Your task is to rewrite texts in Tatar 
to remove any toxic, offensive, or harmful content, while preserving the original meaning, intent, 
and stylistic tone. 

RULES:
1. ALWAYS keep the text in Tatar. Do NOT translate it into Russian, English, or any other language.
2. Remove or neutralize any offensive, rude, or harmful words/phrases.
3. Preserve all factual information and the general meaning.
4. Maintain the natural style, grammar, and flow of Tatar language.
5. NEVER invent new content; only detoxify what is there.
6. NEVER leave the text empty. If the input is fully toxic, output a neutral alternative conveying the intended meaning.

CHAIN-OF-THOUGHT (CoT) APPROACH:
- Step 1: Identify any words, expressions, or structures that are potentially toxic.
- Step 2: Consider alternative neutral expressions in Tatar that convey the same meaning.
- Step 3: Replace toxic elements with neutral equivalents, preserving sentence structure and style.
- Step 4: Ensure the output is grammatically correct and fluent in Tatar.
- Step 5: Double-check that no content is lost and the meaning remains clear.

FEW-SHOT EXAMPLES:

Example 1:
Input: "Бу кеше бөтенләй аңламый, бөтенләй дурак!"
Output: "Бу кеше аңламый, бәлки башкалардан аерыла."

Example 2:
Input: "Син бөтенләй файдасыз кеше!"
Output: "Син үзеңнең юлың белән эшлисең, һәм һәркем үзенчә кирәкле."

Example 3:
Input: "Бу урын чирләвек, бик начар!"
Output: "Бу урын үзенчә катлаулы, ләкин эшләргә мөмкин."

Now, process the following text strictly according to these rules.
"""



with open(JSONL_PATH, "w", encoding="utf-8") as f:
    for text in df['tat_toxic']:
        item = {
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": text}
            ]
        }
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
