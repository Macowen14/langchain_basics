from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


llm = ChatOllama(model="mistral-large-3:675b-cloud", temperature=0.2)
prompt = ChatPromptTemplate.from_template("Explain {topic}")

# This pipe syntax replaces the old LLMChain
# For old llm chain use uv add langchain-community to import LLMChain from langchain.chains
chain = prompt | llm | StrOutputParser()

print(chain.invoke({"topic": "black holes"}))