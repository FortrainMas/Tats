from transformers import AutoModelForSeq2SeqLM, MT5Tokenizer
from transformers import BitsAndBytesConfig

from transformers import AutoTokenizer

model_name = "textdetox/mt5-xl-detox-baseline"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = MT5Tokenizer.from_pretrained("google/mt5-xl")

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto"
)