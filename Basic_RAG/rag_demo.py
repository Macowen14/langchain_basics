
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS

from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --------------------------------------------------
# 1. Setup LLM
# --------------------------------------------------
llm = ChatOllama(
    model="mistral-large-3:675b-cloud", 
    temperature=0.2
)

# --------------------------------------------------
# 2. Load the text file
# --------------------------------------------------
try:
    loader = TextLoader("rag.txt")
    documents = loader.load()
except FileNotFoundError:
    print("Error: 'rag.txt' not found. Please create a dummy file to test.")
    exit()

# --------------------------------------------------
# 3. Split text into chunks
# --------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# --------------------------------------------------
# 4. Create embeddings and vector store
# --------------------------------------------------
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector_store = FAISS.from_documents(docs, embeddings)

# --------------------------------------------------
# 5. Create retriever
# --------------------------------------------------
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# --------------------------------------------------
# 6. Create RAG prompt
# --------------------------------------------------
prompt = ChatPromptTemplate.from_template(
    """
You are an assistant that answers questions using ONLY the provided context.
If the answer is not contained in the context, say "I don't know."

Context:
{context}

Question:
{question}
"""
)

# --------------------------------------------------
# 7. Build RAG chain (LCEL)
# --------------------------------------------------
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# --------------------------------------------------
# 8. Ask a question grounded in rag.txt
# --------------------------------------------------
query = "What is Retrieval-Augmented Generation and why is it useful?"
print(f"\nQuerying: {query}\n")

response = rag_chain.invoke(query)

# --------------------------------------------------
# 9. Output
# --------------------------------------------------
print("Answer:")
print(response)