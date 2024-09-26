import os
import numpy as np
import pandas as pd
import torch
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import faiss
from tqdm import tqdm

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cpu' and torch.backends.mps.is_available():
    device = torch.device('mps')

print(f"Using device: {device}")

model_id = "google/gemma-2b-it"
config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_id, hidden_activation="gelu_pytorch_tanh", token=True)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id, token=True)
llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id, 
                                                 config=config, 
                                                 torch_dtype=torch.float16, 
                                                 low_cpu_mem_usage=False,
                                                 token=True)
llm_model.to(device)

csv_path = "dataset_folder/text_chunks_and_embeddings_df.csv"
text_chunks_and_embedding_df = pd.read_csv(csv_path)

text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

embedding_model = SentenceTransformer("all-mpnet-base-v2").to(device)

embeddings = np.array(text_chunks_and_embedding_df["embedding"].tolist()).astype('float32')
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
if device.type == 'cuda':
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)
index.add(embeddings)

def retrieve_relevant_resources(query: str, n_resources_to_return: int = 5) -> List[Dict]:
    query_embedding = embedding_model.encode(query, convert_to_tensor=True, device=device)
    scores, indices = index.search(query_embedding.cpu().numpy().reshape(1, -1), n_resources_to_return)
    return [{"chunk": pages_and_chunks[i], "score": float(scores[0][j])} for j, i in enumerate(indices[0])]

def prompt_formatter(query: str, context_items: List[Dict]) -> str:
    context = "\n".join([f"[{i+1}] {item['chunk']['sentence_chunk']}" for i, item in enumerate(context_items)])
    base_prompt = f"""Based on the following context items, please answer the query. If the information is not available in the context, please state that you don't have enough information to answer accurately.

Context:
{context}

Query: {query}

Answer: Let's approach this step-by-step:

1) First, I'll identify the key points in the query.
2) Then, I'll search for relevant information in the provided context.
3) Finally, I'll synthesize this information to provide a comprehensive answer.

Here's my response:
"""
    return base_prompt

def ask(query: str, temperature: float = 0.1, max_new_tokens: int = 512) -> str:
    try:
        context_items = retrieve_relevant_resources(query)
        prompt = prompt_formatter(query, context_items)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                temperature=temperature,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.2
            )
        
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = output_text.split("Here's my response:")[-1].strip()
        return answer
    except Exception as e:
        return f"An error occurred: {str(e)}. Please try again or rephrase your question."

if __name__ == "__main__":
    while True:
        query = input("Enter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        response = ask(query)
        print(f"Answer: {response}\n")