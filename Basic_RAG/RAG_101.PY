import os
import sys

# --- 1. SETUP & IMPORTS ---
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---

MODEL_NAME = "mistral-large-3:675b-cloud"


def main():
    print("--- STARTING EXPANDED RAG PIPELINE ---")


    # --- 3. LOAD & SPLIT ---
    loader = TextLoader("expanded_medical_data.txt")
    docs = loader.load()
    
    # We define separators to keep patient records together if possible
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, 
        chunk_overlap=50,
        separators=["[RECORD ID:", "\n\n", "\n", " "]
    )
    splits = text_splitter.split_documents(docs)
    print(f"✅ Split data into {len(splits)} chunks.")

    # --- 4. EMBEDDING ---
    print("⏳ Updating Vector DB...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OllamaEmbeddings(model="mxbai-embed-large"),
        # We save to a different folder to avoid mixing with the previous test
        persist_directory="./chroma_db_expanded"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # Retrieve top 2 matches
    print("✅ Database ready.")

    # --- 5. SETUP CHAIN ---
    llm = ChatOllama(model=MODEL_NAME,temperature=0.2, format="json")
    
    template = """You are a helpful nursing assistant. 
    Answer the question based ONLY on the following patient records:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # --- 6. INTERACTIVE LOOP ---
    print("\n" + "="*40)
    print("🤖 SYSTEM READY. Type 'exit' to quit.")
    print("Try asking: 'What does Michael have?' or 'Who broke their wrist?'")
    print("="*40 + "\n")

    while True:
        try:
            user_input = input("Enter Question: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            print("Thinking...")
            response = rag_chain.invoke(user_input)
            print(f"\n>> ANSWER: {response}\n")
            print("-" * 20)
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()