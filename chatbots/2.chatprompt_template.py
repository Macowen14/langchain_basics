from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {domain} expert."),
    ("human", "Explain the {subject}."),
])

prompt = chat_template.invoke({
    "domain": "Ai Engineering",
    "subject": "LangChain",
})

result = model.invoke(prompt)

print(result.content)
