# NLLB-200-distilled-600M-LoRA for Russian — Komi-Zyrian translation

This is a LoRA adapter on top of `facebook/nllb-200-distilled-600M` for bidirectional translation between Russian and Komi-Zyrian.

It was trained on 50,815 parallel sentence pairs from `Horeknad/komi-russian-parallel-corpora`. The data was cleaned, deduplicated, filtered by length, and split into train/validation/test sets. Both translation directions were included during training. It was fine-tuned using LoRA with 4-bit NF4 quantization.

## Evaluation

Test set SacreBLEU: **25.14**

That is a solid result for a Komi-Russian model trained on a relatively small parallel corpus. For context, published Komi-related benchmarks are still modest: a 2023 NLLB-based Finno-Ugric MT paper reports **26.4 BLEU** for high-resource Komi on FLORES, and a 2024 SMUGRI paper reports Komi-related RU-KPV scores of **23.4 / 17.3** for its best translation-tuned model, versus **6.7 / 0.5** for GPT-3.5-turbo.

## Inference

Get the original NLLB model and the adapter.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from peft import PeftModel

base_model = "facebook/nllb-200-distilled-600M"
adapter_id = "pymlex/nllb-600M-kpv-rus"

tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    base_model,
    quantization_config=quantization_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(model, adapter_id)
model.eval()
```

Use this function for translation.

```python
def translate(text, src_lang, tgt_lang, max_new_tokens=128, num_beams=4):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

# Language codes:
# Russian:    "rus_Cyrl"
# Komi-Zyrian:"kpv_Cyrl"

print(translate("Бесплатный русско-коми переводчик и словарь.", "rus_Cyrl", "kpv_Cyrl"))
print(translate("Кывъяс, кывтэчасъяс да сёрникузяяс.", "kpv_Cyrl", "rus_Cyrl"))
````
