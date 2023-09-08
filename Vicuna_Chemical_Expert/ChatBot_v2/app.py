from dataclasses import dataclass
from typing import Literal
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TextStreamer
from peft import prepare_model_for_kbit_training
import transformers
import torch
from torch.utils.data import DataLoader, Dataset
from peft import PeftModel, PeftConfig, LoraConfig
import streamlit as st
import  streamlit_toggle as tog
from contextlib import contextmanager, redirect_stdout
from langchain import PromptTemplate, HuggingFacePipeline, LLMChain
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from io import StringIO
from chroma_settings import CHROMA_SETTINGS
from tokens import HUGGINGFACEHUB_API_TOKEN
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN


# Redirect stdout to streamlit
@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            std = stdout.getvalue()
            if st.session_state["vectordb"]: 
                 index = std.find("Helpful Answer:") 
            else: 
                 index = len(prompt)+5
            output_func(std[index:-5])
            return ret
        
        stdout.write = new_write
        yield

# App title
st.set_page_config(page_title="Llama 2 Chatbot")

# Store LLM model case
if "model_case" not in st.session_state:
    st.session_state["model_case"] = 'chemical'
    st.session_state.model_path = 'FelixChao/vicuna-7B-chemical'
    st.session_state.avatar = "https://images.squarespace-cdn.com/content/v1/6213c340453c3f502425776e/4d53dea3-b793-4fd8-ad14-2d616d894eb4/Stable+Vicuna_clipdrop-enhance.png"

@st.cache_resource()
def get_model(model_path):
  model_id = model_path #model path in huggingface
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  tokenizer.pad_token = tokenizer.eos_token
  model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16 , device_map="auto", trust_remote_code=True)
  model.config.use_cache = True
  model.eval()

  return model , tokenizer

def clear_chat_history():
    st.session_state.messages = [{"role": st.session_state['model_case'],"avatar":st.session_state.avatar, "content": f"What {st.session_state['model_case']} questions do you want to ask today?"}]

def selectbox_call():
    option = st.session_state['model']

    if option == 'Chemistry':
       st.session_state['model_case'] = 'chemical'
       st.session_state.model_path = 'FelixChao/vicuna-7B-chemical'
       st.session_state.avatar = "https://images.squarespace-cdn.com/content/v1/6213c340453c3f502425776e/4d53dea3-b793-4fd8-ad14-2d616d894eb4/Stable+Vicuna_clipdrop-enhance.png"
    elif option == 'Physics':
       st.session_state['model_case'] = 'physics'
       st.session_state.model_path = 'FelixChao/vicuna-7B-physics'
       st.session_state.avatar = "https://lmsys.org/images/blog/vicuna/vicuna.jpeg"
    elif option == 'Mathematics':
       st.session_state['model_case'] = 'math'
       st.session_state.model_path = 'FelixChao/llama2-13b-math1.2'
       st.session_state.avatar = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRgAdorTGaoiKdZLaAzBWN5H5YogEJ837LDJQ&usqp=CAU"

    st.session_state.messages.append({"role": st.session_state['model_case'],"avatar":st.session_state.avatar, "content": f"What {st.session_state['model_case']} questions do you want to ask today?"})
    # clear_chat_history()


with st.sidebar:
    st.title('?? Vicuna Chatbot')

    st.subheader('Models')

    
    option = st.sidebar.selectbox(
        'Choose a model:',
        ('Chemistry','Physics','Mathematics'),
        on_change = selectbox_call,
        key = 'model',
    )

    st.divider()

    st.subheader('Parameters')
    
    temperature = st.sidebar.slider('temperature', min_value=0.001, max_value=2.000, value=0.001, step=0.001)
    top_k = st.sidebar.slider('top_k', min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=64, max_value=4096, value=2048, step=8)
    # st.write(switch)

    # txt = st.sidebar.text_area("Prompt Template", "Let's think step by step.")
    # st.write(txt)
    chat_model , tokenizer = get_model(model_path=st.session_state.model_path)
    
    st.divider()

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "chemical","avatar":"https://images.squarespace-cdn.com/content/v1/6213c340453c3f502425776e/4d53dea3-b793-4fd8-ad14-2d616d894eb4/Stable+Vicuna_clipdrop-enhance.png", "content": f"What {st.session_state['model_case']} questions do you want to ask today?"}]


# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"],avatar=message["avatar"]):
        st.write(message["content"])




with st.sidebar:
     cols1, cols2 = st.columns(2)

     with cols2:
          switch = st.toggle(label="Vectordb",value=False,key="vectordb")
     with cols1:
          st.button('Clear Chat History', on_click=clear_chat_history, type="secondary")

@st.cache_resource()
def initialize_vectordb(reference):
    loader = DirectoryLoader(f'./{reference}', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    #splitting the text into
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    instructor_embeddings =  SentenceTransformerEmbeddings(model_name = "all-MiniLM-L6-v2")
    # Embed and store the texts
    # Supplying a persist_directory will store the embeddings on disk
    persist_directory = 'db'

    ## Here is the nmew embeddings being used
    embedding = instructor_embeddings

    vectordb = Chroma.from_documents(documents=texts,
                                     embedding=embedding,
                                     persist_directory=persist_directory,
                                     client_settings=CHROMA_SETTINGS
                                     )
    vectordb.persist()
    vectordb=None

    return embedding, persist_directory

def get_vectordb_retriever():
    embedding , persist_directory = initialize_vectordb(reference="Hydrogen")
    db = Chroma(
                 embedding_function=embedding,
                 persist_directory=persist_directory,
                 client_settings=CHROMA_SETTINGS
                )
    retriever = db.as_retriever(search_kwargs={'k': 4})
    return retriever



# Function for generating LLaMA2 response
def generate_llama2_response(prompt_input):
    encoding = tokenizer(prompt_input, return_tensors="pt").to('cuda')
    streamer = TextStreamer(tokenizer)

    if(st.session_state['vectordb']):
       pipe = pipeline(
         "text-generation", model=chat_model, tokenizer=tokenizer, max_new_tokens=max_length, do_sample=True, temperature=temperature, top_k = top_k, streamer=streamer
       )
       llm = HuggingFacePipeline(pipeline=pipe)
       retriever = get_vectordb_retriever()
       qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False)
       predict = qa(prompt_input)["result"]
    else:
        output = chat_model.generate(input_ids=encoding.input_ids,streamer=streamer, attention_mask=encoding.attention_mask, max_new_tokens=max_length, do_sample=True, temperature=temperature, eos_token_id=tokenizer.eos_token_id, top_k = top_k)
        predict = tokenizer.decode(output[0], skip_special_tokens=True)
        predict = predict[len(prompt_input)+1:]

    return predict



# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user","avatar":"https://www.shareicon.net/data/128x128/2016/07/05/791219_man_512x512.png", "content": prompt})
    with st.chat_message("user",avatar="https://www.shareicon.net/data/128x128/2016/07/05/791219_man_512x512.png"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message(st.session_state["model_case"],avatar=st.session_state.avatar):
           placeholder = st.empty()
           with st_capture(placeholder.markdown):
            response = generate_llama2_response(prompt)
    message = {"role": st.session_state["model_case"],"avatar":st.session_state.avatar,"content": response}
    st.session_state.messages.append(message)