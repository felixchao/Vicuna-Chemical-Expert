from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling, TextStreamer

tokenizer = AutoTokenizer.from_pretrained("FelixChao/vicuna-7B-chemical")
model = AutoModelForCausalLM.from_pretrained("FelixChao/vicuna-7B-chemical",device_map="auto")
streamer = TextStreamer(tokenizer,skip_prompt=True,skip_special_token=True)


def generate(index):

  example_text = "Who are you?"
  

  print("Question:")
  print(example_text)

  encoding = tokenizer(example_text, return_tensors="pt").to("cuda:0")
  output = model.generate(input_ids=encoding.input_ids,streamer=streamer,attention_mask=encoding.attention_mask, max_new_tokens=512, do_sample=True, eos_token_id=tokenizer.eos_token_id)
  predict = tokenizer.decode(output[0], skip_special_tokens=True)
  predict = predict[len(example_text)+1:]


  print()
  
generate(0)