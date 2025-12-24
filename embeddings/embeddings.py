# from langchain_huggingface import HuggingFaceEmbeddings # Commented until downloaded
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()

# --- USING OLLAMA ---
# Note: mxbai-embed-large performs best when you tell it the task
embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

# --- USING HUGGING FACE (Future) ---
# After downloading, swap the block above with this:
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "Nairobi is capital of Kenya"

documents = [
    "Delhi is the capital of India",
    "Washington is the capital of USA",
    "Paris is the capital of France"
]

# Generate embedding for a single query
result = embeddings.embed_query(text)

print(f"Embedding Length: {len(result)}")
print(f"First 5 dimensions: {result[:5]}")

# If you want to see how it relates to the documents:
doc_results = embeddings.embed_documents(documents)
print(f"Generated embeddings for {len(doc_results)} documents.")