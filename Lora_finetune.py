from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TextStreamer
import transformers
import torch
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import AutoTokenizer, pipeline
from peft import prepare_model_for_kbit_training



model_id = "lmsys/vicuna-13b-v1.5"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)

from peft import LoraConfig, get_peft_model

config = LoraConfig(
     r=16,
     lora_alpha=32,
     lora_dropout=0.05,
     bias="none",
     task_type="CAUSAL_LM",
     inference_mode=False,
 )


model = get_peft_model(model, config)
model.print_trainable_parameters()


from datasets import load_dataset

data = load_dataset("andersonbcdefg/chemistry")
tokenizer.pad_token = tokenizer.eos_token



train_dataset = data['train'].map(lambda x: {"input_text": f"### Instruction: {x['instruction']}\n### Output: {x['output']}"})

# Tokenize the datasets
train_encodings = tokenizer(train_dataset['input_text'], truncation=True, padding=True, max_length=256, return_tensors='pt')

class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = item["input_ids"].clone()
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
        
# Convert the encodings to PyTorch datasets
train_dataset = TextDataset(train_encodings)


trainer = transformers.Trainer(
    model=model,
    train_dataset=train_dataset,
    # eval_dataset=val_dataset,
    args=transformers.TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        max_steps=1000,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        lr_scheduler_type='cosine',
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

model.save_pretrained("lora_adapter")

        
