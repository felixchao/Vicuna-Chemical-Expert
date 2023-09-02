# Vicuna-Chemical-Expert
![vicuna_chemist_logo](https://github.com/felixchao/Vicuna-Chemical-Expert/assets/75468071/785af3da-2fa7-4619-b72c-088dcd06eeb6)



## Update Logs
* 2023.8.15: Create Chatbot_v2
  
  * Add features: **Langchain, ChromaDB(VetcorDB)**
  * Toggle switch for searching **Hydrogen paper**
  
* 2023.7.30: Create Chatbot_v1 

   * Adding Multiple Models: **Chemical, Physics, Mathematics**
   * Create Streamlit app
## Introduction
This is the repo for Vicuna Chemical Expert, which can help to solve some chemical questions. This model was finetuned by the **sharded version** of **lmsys/vicuna-7b-v1.3**, and it can be trained on **4x V100 32GB**.

## Finetune
* Use **Qlora tuning** in Peft
* Vicuna 7B was finetuned based on **chemistry** and **chemical industry** domain.
* **Parameters** available in training

| trainable params | all params | trainable% |
|:----------------:|:----------:|:----------:|
|     13107200     | 6685086720 |   0.1961   |
* When training is done, **merge lora** back to base model
* Below is the finetuning train/loss graph:
![finetune_process](https://github.com/felixchao/Vicuna-Chemical-Expert/assets/75468071/f61b5b84-1217-4110-8889-8e434f9dc2d2)

* **HuggingFace**: [FelixChao/vicuna-7B-chemical](https://huggingface.co/FelixChao/vicuna-7B-chemical)

## Setup
To inference this model on your local

* Create conda environment and activate
```sh
conda create -n vicuna-chemical python=3.10 
conda activate vicuna-chemical
```
* Download Chatbot_v1 or Chatbot_v2
* Install dependencies
```sh
pip install -r requirements.txt
```
* Run streamlit app
```sh
streamlit run app.py
```

Note: Please make sure that your gpu RAM (**at least 16GB**) is enough for loading model, avoid CUDA Out Of Memory(**OOM**).

## Usage



