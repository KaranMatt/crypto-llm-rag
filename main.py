import torch
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer,AutoModelForCausalLM,pipeline
from sentence_transformers import CrossEncoder
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

class QuestionResponse(BaseModel):
    question:str

class ResponseClass(BaseModel):
    question:str
    response:str

embeddings=None
pipe=None
vector_db=None
rerank=None

@asynccontextmanager
async def lifespan(app:FastAPI):
    global embeddings,pipe,vector_db,rerank

    embeddings=HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
    print('Embeddings Loaded')
    vector_db=FAISS.load_local('Vector DB Index',embeddings=embeddings,allow_dangerous_deserialization=True)
    print('Vector DB Loaded')
    MODEL='Qwen/Qwen2.5-1.5B-Instruct'
    tokenizer=AutoTokenizer.from_pretrained(MODEL)
    model=AutoModelForCausalLM.from_pretrained(MODEL,device_map='auto',dtype=torch.bfloat16,low_cpu_mem_usage=True)
    pipe=pipeline(task='text-generation',temperature=0.4,do_sample=True,tokenizer=tokenizer,model=model,max_new_tokens=512,repetition_penalty=1.1,
              no_repeat_ngram_size=3)
    print('LLM Pipeline Loaded')
    rerank=CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')
    print('Reranker Model Loaded')
    print('Models Loaded!!!')

    yield
    print('Shutdown Initiated')
    vector_db=None
    pipe=None
    rerank=None
    pipe=None

app=FastAPI(title='Crypto LLM RAG API',lifespan=lifespan)

@app.get('/root')
def root():
    return {'message':'Welcome to the RAG API'}

@app.get('/health')
def health():
    if vector_db and pipe:
        return {'status':'Active','Models Loaded':True}
    else:
        return {'status':'Not yet Active','Models Loaded':False}

@app.post('/ask',response_model=ResponseClass)
def predict(request:QuestionResponse):
    initial_search=vector_db.similarity_search(query=request.question,k=25)
    pairs=[[request.question,doc.page_content] for doc in initial_search]
    scores=rerank.predict(pairs)
    scored_results=sorted(zip(scores,initial_search),key=lambda x:x[0],reverse=True)
    final_search=[doc for score,doc in scored_results[:5]]
    context_list=[]

    for doc in final_search:
        file_path=doc.metadata.get('source','unknown')
        filename=os.path.basename(file_path)
        page_num=doc.metadata.get('page',0)+1
        header=f'[Doc:{filename} | Page:{page_num}]'
        context_list.append(f'{header}\n{doc.page_content}')
    context='\n\n-\n\n'.join(context_list)
    prompt=f'''You are an Experienced Financial Analyst.Your Job is to answer to the question using the CONTEXT provided
         CRITICAL:You must Always cite the source in the format [Doc:filename | Page:X]  Example: [Doc:XYZ.pdf | Page:23] 
         
         IMPORTANT RULES:
         1.The answer must not exceed 500 Words
         2.You must always cite the sources as instructed in the format [Doc:filename | Page:X] 
         3.You must always stick to the information provied in the document 
         4.Make sure you are factually accurate
         5.If the information is not present then say 'No information is present in the document' 
         
         Question:{request.question}

         Context:{context}

         Answer:'''    
        
    response=pipe(prompt,return_full_text=False)
    answer=response[0]['generated_text']
    return ResponseClass(question=request.question,response=answer)

