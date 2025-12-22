from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from typing import TypedDict, Optional, Literal, List
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOllama(
    model="mistral-large-3:675b-cloud",
    temperature=0.2,
    format="json"
)

class Review(BaseModel):
    key_themes:List[str] = Field(description="List all important themes and aspects of the product")
    summary: str = Field(description = "Write a very short one-line summary")
    review: str = Field(description = "Write a detailed and informative review")
    pros: Optional[list[str]] = Field (description= "List the main advantages of the product")
    cons: Optional[list[str]] = Field (description= "List the main disadvantages of the product")
    
structured_model = model.with_structured_output(Review)

prompt = """
Write a product review for the Dell G3 15 3590 Gaming Laptop.
Return a JSON object with the following keys at the root: 
"key_themes", "summary", "review", "pros", and "cons".
Do not wrap the response in a "product_review" or "data" key.
"""
result = structured_model.invoke(prompt)

print(result)
print(type(result))