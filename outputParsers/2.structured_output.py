from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser 
from langchain_ollama import ChatOllama

# 1. Setup the Model
model = ChatOllama(model="mistral-large-3:675b-cloud", format="json")

# 2. Define Schema
class ResearchPaper(BaseModel):
    title: str = Field(description="A catchy title for the topic")
    summary: str = Field(description="A 2-sentence summary")
    tags: List[str] = Field(description="List of 3 relevant keywords")

# 3. Setup Base Parser and the Fixer
parser = PydanticOutputParser(pydantic_object=ResearchPaper)
# You pass the LLM into the fixer so it knows which model to use for repairs
fix_parser = OutputFixingParser.from_llm(parser=parser, llm=model)

# 4. Prepare Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant.\n{format_instructions}"),
    ("human", "Research the topic of {topic}")
])

prompt_with_instructions = prompt.partial(format_instructions=parser.get_format_instructions())

# 5. The Chain (Using fix_parser instead of parser)
chain = prompt_with_instructions | model | fix_parser

result = chain.invoke({"topic": "Quantum Mechanics"})

print(result.title)