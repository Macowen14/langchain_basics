from langchain_core.tools import tool


@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiplies two numbers."""
    return a * b

result = multiply_numbers.invoke({'a': 3, 'b': 4})
print(f"The result of multiplication is: {result}")


print(f"Tool args: {multiply_numbers.args}")
print(f"Tool description: {multiply_numbers.description}")