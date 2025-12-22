from langchain_google_genai import ChatGoogleGenerativeAI
import dotenv
from typing import TypedDict, Optional
from typing_extensions import Annotated

dotenv.load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

class Review(TypedDict):
    key_themes: Annotated[list[str], "List all important themes and aspects of the product"]
    summary: Annotated[str, "Write a very short one-line summary"]
    review: Annotated[str, "Write a detailed and informative review"]
    pros: Annotated[Optional[list[str]], "List the main advantages of the product"]
    cons: Annotated[Optional[list[str]], "List the main disadvantages of the product"]

structured_model = model.with_structured_output(Review)

prompt = """
Write a concise product review.

Product: Apple MacBook Pro M1
"""

result = structured_model.invoke(prompt)

print(result)
print(type(result))
