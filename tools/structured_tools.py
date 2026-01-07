from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class AddNumbersInput(BaseModel):
    a: float = Field(..., description="The first number to add.")
    b: float = Field(..., description="The second number to add.")

def add_numbers(a:float, b:float) -> float:
    """Adds two numbers."""
    return a + b


add_numbers_tool = StructuredTool.from_function(
    func=add_numbers,
    name="add_numbers",
    description="Adds two numbers together.",
    args_schema = AddNumbersInput,
)

result = add_numbers_tool.invoke({'a': 5, 'b': 7})
print(f"The result of addition is: {result}")
print(f"Tool args: {add_numbers_tool.args}")
print(f"Tool description: {add_numbers_tool.description}")