from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

def load_chat_history_from_txt(path):
    messages = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Human:"):
                    messages.append(
                        HumanMessage(content=line.replace("Human:", "").strip())
                    )
                elif line.startswith("AI:"):
                    messages.append(
                        AIMessage(content=line.replace("AI:", "").strip())
                    )
    except FileNotFoundError:
        pass
    return messages

def save_message_to_txt(path, role, content):
    with open(path, "a") as f:
        f.write(f"{role}: {content}\n")

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support agent."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}"),
])

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1
)

CHAT_FILE = "chatbot_history.txt"

while True:
    user_input = input("User: ").strip()
    if user_input.lower() == "quit":
        break

    chat_history = load_chat_history_from_txt(CHAT_FILE)

    prompt = chat_template.invoke({
        "chat_history": chat_history,
        "user_input": user_input
    })

    result = model.invoke(prompt)

    print("Assistant:", result.content)

    # 🔐 MEMORY WRITE-BACK
    save_message_to_txt(CHAT_FILE, "Human", user_input)
    save_message_to_txt(CHAT_FILE, "AI", result.content)
