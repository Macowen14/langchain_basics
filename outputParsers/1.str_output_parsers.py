from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# The "Smart" model for deep understanding
smart_model = ChatOllama(model="mistral-large-3:675b-cloud", temperature=0.2)

# The "Fast" model for the quick summary
fast_model = ChatOllama(model="gemma3:27b-cloud", temperature=0)

template1 = PromptTemplate.from_template("What is {subject} about?")
template2 = PromptTemplate.from_template("Write summary on {subject}")

parser = StrOutputParser()

chain = template1 | smart_model | parser | template2 | fast_model | parser
final_output = chain.invoke({"subject": "LangChain"})

print(final_output)