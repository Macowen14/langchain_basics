from typing import List
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Setup Model
model = ChatOllama(model="mistral-large-3:675b-cloud", temperature=0.2, format="json")

# 2. Define Schemas
class ResearchPaper(BaseModel):
    title: str = Field(description="A catchy title")
    summary: str = Field(description="A 2-sentence summary")
    tags: List[str] = Field(description="3 keywords")

class RefinedResearch(BaseModel):
    title: str
    summary: str
    tags: List[str]
    critique: str = Field(description="What was improved from the original?")

# 3. Create Parsers
parser_1 = PydanticOutputParser(pydantic_object=ResearchPaper)
parser_2 = PydanticOutputParser(pydantic_object=RefinedResearch)

# 4. Define Prompt 1: Initial Research
prompt_1 = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant.\n{format_instructions}"),
    ("human", "Research the topic of {topic}")
]).partial(format_instructions=parser_1.get_format_instructions())

# 5. Define Prompt 2: Refinement
prompt_2 = ChatPromptTemplate.from_messages([
    ("system", "You are a senior editor specializing in {topic}.\n{format_instructions}"),
    ("human", """
    The user wanted research on: {topic}
    
    Here is the draft produced by the junior assistant:
    {original_research}
    
    Please refine this. Ensure the title is punchy and the summary accurately 
    reflects the core principles of {topic}. Provide a critique of what was missing.
    """)
]).partial(format_instructions=parser_2.get_format_instructions())

# 6. Build the Sequential Chain
# Chain 1: Research
chain_1 = prompt_1 | model | parser_1

# Chain 2: Refine (We use a lambda or dict to map the previous output)
full_chain = (
    {"original_research": chain_1, "topic": RunnablePassthrough()} 
    | prompt_2 
    | model 
    | parser_2
)

# 7. Execute
result = full_chain.invoke({"topic": "Quantum Mechanics"})

print(f"Refined Title: {result.title}")
print(f"Critique: {result.critique}")