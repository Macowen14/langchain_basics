from langchain_google_genai import ChatGoogleGenerativeAI
import dotenv
from typing import TypedDict

dotenv.load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

class Review(TypedDict):
    summary: str
    review: str

structured_model = model.with_structured_output(Review)

prompt = """
Write a concise product review.

Product: Apple Macbook Pro M1
"""

result = structured_model.invoke(prompt)

print(result)
print(type(result))
