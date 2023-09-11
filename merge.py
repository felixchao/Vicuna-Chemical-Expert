from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling, TextStreamer

tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.5")
model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-13b-v1.5",device_map="auto")

from peft import PeftModel, PeftConfig, LoraConfig

config = LoraConfig.from_pretrained("lora_adpater")


# load perf model with new adapters
model = PeftModel.from_pretrained(
    model,
    "lora_adpater",
)

merged_model = model.merge_and_unload()

merged_model.push_to_hub("Vicuna-Chemical-Expert")
tokenizer.push_to_hub("Vicuna-Chemical-Expert")