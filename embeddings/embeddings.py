# from langchain_huggingface import HuggingFaceEmbeddings Future
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
load_dotenv()

# --- USING OLLAMA ---
# Note: mxbai-embed-large performs best when you tell it the task
embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

# --- USING HUGGING FACE (Future) ---
# After downloading, swap the block above with this:
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "Nairobi is the capital of Kenya"

documents = [
    "Delhi is the capital of India",
    "Washington is the capital of USA",
    "Paris is the capital of France",
    "Nairobi is the capital of Kenya",
    "Tokyo is the capital of Japan",
    "I love pizza"
]

# Generate embedding for a single query
result = embeddings.embed_query(text)

print(f"Embedding Length: {len(result)}")
print(f"First 5 dimensions: {result[:5]}")

# Generate embeddings for multiple documents
doc_results = embeddings.embed_documents(documents)
print(f"Generated embeddings for {len(doc_results)} documents.")

similarity_score = cosine_similarity([result], doc_results)
print(f"Similarity score: {similarity_score}")