import os
import numpy as np
import pandas as pd
import textwrap
import torch
from typing import List, Dict
from sentence_transformers import util, SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

import warnings
warnings.filterwarnings("ignore", message="`clean_up_tokenization_spaces` was not set")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

csv_path = "dataset_folder/text_chunks_and_embeddings_df.csv"
text_chunks_and_embedding_df = pd.read_csv(csv_path)

text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")
embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to(device)

embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device=device)

def generate_pseudo_context(chunk: str, all_chunks: List[str]) -> str:
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_chunks)
    chunk_vector = vectorizer.transform([chunk])
    similarities = cosine_similarity(chunk_vector, tfidf_matrix)[0]
    top_similar_indices = similarities.argsort()[-4:-1][::-1]
    key_terms = set()
    for idx in top_similar_indices:
        terms = vectorizer.get_feature_names_out()[tfidf_matrix[idx].nonzero()[1]]
        key_terms.update(terms[:5])
    context = f"This chunk discusses nutrition topics related to: {', '.join(list(key_terms)[:10])}"
    return context

def create_pseudo_contextual_embeddings(chunks: List[Dict], batch_size: int = 1000) -> List[Dict]:
    all_chunks = [chunk['sentence_chunk'] for chunk in chunks]
    for i in tqdm(range(0, len(chunks), batch_size), desc="Creating pseudo-contextual embeddings"):
        batch = chunks[i:i+batch_size]
        for chunk in batch:
            context = generate_pseudo_context(chunk['sentence_chunk'], all_chunks)
            contextualized_chunk = f"{context} {chunk['sentence_chunk']}"
            chunk['contextual_embedding'] = embedding_model.encode(contextualized_chunk, convert_to_tensor=True, device=device)
    return chunks

def bm25_ranking(query: str, chunks: List[Dict], k: int = 20) -> List[Dict]:
    tokenized_corpus = [chunk['sentence_chunk'].lower().split() for chunk in chunks]
    tokenized_query = query.lower().split()
    bm25 = BM25Okapi(tokenized_corpus)
    doc_scores = bm25.get_scores(tokenized_query)
    top_k_indices = np.argsort(doc_scores)[-k:][::-1]
    return [chunks[i] for i in top_k_indices]

def retrieve_relevant_resources(query: str, chunks: List[Dict], embeddings: torch.tensor, model: SentenceTransformer=embedding_model, n_resources_to_return: int=20):
    query_embedding = model.encode(query, convert_to_tensor=True, device=device)
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    scores, indices = torch.topk(input=dot_scores, k=n_resources_to_return)
    embedding_results = [chunks[i] for i in indices]
    bm25_results = bm25_ranking(query, chunks, k=n_resources_to_return)
    combined_results = list({chunk['sentence_chunk']: chunk for chunk in embedding_results + bm25_results}.values())
    combined_results.sort(key=lambda x: util.dot_score(query_embedding, x['contextual_embedding'])[0], reverse=True)
    return combined_results[:n_resources_to_return]

def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

def prompt_formatter(query: str, context_items: List[Dict]) -> str:
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])
    base_prompt = """You are a knowledgeable and helpful dietitian chatbot. Your purpose is to provide accurate and helpful information about nutrition and diet based on the context provided. Always strive to give comprehensive and well-explained answers.

Based on the following context items, please answer the query.
If the context doesn't provide enough information to fully answer the query, use your general knowledge about nutrition to supplement the answer, but prioritize the information from the context.
Make sure your answers are as in-depth as possible while remaining relevant to the query.

Context:
{context}

User query: {query}

Answer:"""
    return base_prompt.format(context=context, query=query)

def ask(query, temperature=0.7, max_new_tokens=512):
    context_items = retrieve_relevant_resources(query=query, chunks=pages_and_chunks, embeddings=embeddings)
    prompt = prompt_formatter(query=query, context_items=context_items)
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = llm_model.generate(**input_ids, temperature=temperature, do_sample=True, max_new_tokens=max_new_tokens)
    output_text = tokenizer.decode(outputs[0])
    output_text = output_text.replace(prompt, "").strip()
    output_text = output_text.replace("<bos>", "").replace("<eos>", "").strip()
    return output_text

model_id = "google/gemma-2b-it"
config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_id, hidden_activation="gelu_pytorch_tanh", token=True)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id, token=True)
llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id, 
                                                 config=config, 
                                                 torch_dtype=torch.float16, 
                                                 low_cpu_mem_usage=False,
                                                 token=True)
llm_model.to(device)

pages_and_chunks = create_pseudo_contextual_embeddings(pages_and_chunks)
embeddings = torch.stack([chunk['contextual_embedding'] for chunk in pages_and_chunks]).to(device)

def main():
    while True:
        query = input("Enter your nutrition-related question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        answer = ask(query)
        print(f"\nDietitian Chatbot: {answer}\n")

if __name__ == "__main__":
    main()