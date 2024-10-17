from transformers import pipeline
import torch

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct")
pipe(messages)

from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def chat_with_model(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Chat loop
print("Chat with the model. Type 'quit' to exit.")
chat_history = ""
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    
    chat_history += f"Human: {user_input}\n"
    prompt = chat_history + "AI:"
    response = chat_with_model(prompt)
    chat_history += f"AI: {response}\n"
    print("AI:", response)
