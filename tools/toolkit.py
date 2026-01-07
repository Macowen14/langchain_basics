from langchain_core.tools import tool

@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiplies two numbers."""
    return a * b

@tool 
def divide_numbers(a: float, b: float) -> float:
    """Divides the first number by the second number."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

@tool
def subtract_numbers(a: float, b: float) -> float:
    """Subtracts the second number from the first number."""
    return a - b


@tool
def power_numbers(a: float, b: float) -> float:
    """Raises the first number to the power of the second number."""
    return a ** b


class MathTools:

    def get_tools(self):
        return [
            multiply_numbers,
            divide_numbers,
            subtract_numbers,
            power_numbers,
        ]
   

toolkit = MathTools()
tools = toolkit.get_tools()

for t in tools:
    result = t.invoke({'a': 8, 'b': 2})
    print(f"The result of {t.name} is: {result}")
    print(f"Tool args: {t.args}")
    print(f"Tool description: {t.description}")