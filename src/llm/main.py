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
model = AutoModelForCausalLM.from_pretrained(
    gen_model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True,
)
model = model.to(device)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

def generate_response(user_input):
    context = search_context(user_input)
    prompt = """
    **Role & Objective:**  
    You are an expert document analyst and AI assistant. Your task is to analyze the given document and provide an accurate, well-structured, and comprehensive answer to the specified question. Your response should be fact-based, reference relevant sections of the document, and follow the provided guidelines.

    **Document:**  
    {context}  

    **Question:**  
    {user_input}

    ### Response Guidelines:
    1. **Structure & Clarity:**
    - Begin with a summary of the relevant part(s) of the document.  
    - Provide a detailed explanation answering the question directly.  
    - If necessary, include examples or case-specific details.  

    2. **Quoting & Referencing:**
    - Cite exact excerpts from the document where applicable.  
    - Indicate section/page numbers if available.  

    3. **Formatting:**
    - Use bullet points or subheadings for better readability.  
    - Maintain a professional and neutral tone.  

    4. **Additional Insights (if applicable):**
    - If the document lacks information to fully answer the question, state that explicitly and suggest possible interpretations or external references.  

    5. **Word Limit:**  
    - Aim for a response between [X] and [Y] words depending on complexity.  

    **Example Response Format:**

    ### Summary of the Relevant Section
    (Provide a concise summary of the documents content related to the question.)

    ### Detailed Answer
    (Provide a structured, well-explained response, citing relevant text when necessary.)

    ### Key References
    (Include citations, page numbers, or section names.)
    """
    
    output = generator(
        prompt,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1
    )
    
    if isinstance(output, list) and len(output) > 0:
        generated_text = output[0]['generated_text']
        answer_parts = generated_text.split("Answer:")
        if len(answer_parts) > 1:
            return answer_parts[-1].strip()
        return generated_text.strip()
    
    return "I apologize, but I couldn't generate a proper response."


if __name__ == '__main__':
    print("Chatbot ready. Type 'exit' to quit.")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        chatbot_response = generate_response(user_input)
        print("Chatbot:", chatbot_response)