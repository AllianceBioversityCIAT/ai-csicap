from pathlib import Path
import lancedb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = str(BASE_DIR / "db" / "csicapdb")
TABLE_NAME = "files"

db = lancedb.connect(DB_PATH)
try:
    table = db.open_table(TABLE_NAME)
except Exception:
    table = db.create_table(TABLE_NAME, schema=schema)

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(embedding_model_name)

def get_embedding(text):
    return embedding_model.encode(text)

def search_context(query, top_k=3):
    query_embedding = get_embedding(query)
    results = table.search(query_embedding.tolist(), vector_column_name="vector").limit(top_k).to_list()
    context_list = [res["title"] for res in results if "title" in res]
    context = " ".join(context_list)
    return context

gen_model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
model = AutoModelForCausalLM.from_pretrained(gen_model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cpu")

def generate_response(user_input):
    context = search_context(user_input)
    messages = [
        {"role": "system", "content": f"Use this context to answer: {context}"},
        {"role": "user", "content": user_input}
    ]
    output = generator(messages, max_new_tokens=150, do_sample=True, temperature=0.7)
    response = output[0]["generated_text"]
    return response.split("user")[-1].strip()

if __name__ == '__main__':
    print("Chatbot ready. Type 'exit' to quit.")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        chatbot_response = generate_response(user_input)
        print("Chatbot:", chatbot_response)