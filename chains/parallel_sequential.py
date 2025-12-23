from typing import List
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from pydantic import BaseModel, Field
import textwrap

# 1. Setup Model l
model = ChatOllama(model="gemma3:27b-cloud", temperature=0.1, format="json")

# 2. Define Pydantic Schemas
class MarketingPitch(BaseModel):
    tagline: str = Field(description="A catchy 1-sentence tagline")
    key_benefits: List[str] = Field(description="List of 3 user benefits")

class TechSpec(BaseModel):
    requirements: List[str] = Field(description="3 hardware or software requirements")
    complexity: str = Field(description="Low, Medium, or High")

class LaunchPlan(BaseModel):
    product_name: str
    marketing_summary: str
    tech_summary: str
    strategy: str = Field(description="A 2-sentence go-to-market strategy")

# 3. Create Parsers
marketing_parser = PydanticOutputParser(pydantic_object=MarketingPitch)
tech_parser = PydanticOutputParser(pydantic_object=TechSpec)
final_parser = PydanticOutputParser(pydantic_object=LaunchPlan)

# 4. Define Specialized Chains
marketing_prompt = ChatPromptTemplate.from_template(
    "You are a marketing expert. Create a pitch for: {product_name}\n{format_instructions}"
).partial(format_instructions=marketing_parser.get_format_instructions())

tech_prompt = ChatPromptTemplate.from_template(
    "You are a CTO. Create technical specs for: {product_name}\n{format_instructions}"
).partial(format_instructions=tech_parser.get_format_instructions())

marketing_chain = marketing_prompt | model | marketing_parser
tech_chain = tech_prompt | model | tech_parser

# 5. PARALLEL STEP: Run both specialists at once
# We also use RunnablePassthrough to keep the product_name for the next step
parallel_brainstorm = RunnableParallel(
    marketing=marketing_chain,
    tech=tech_chain,
    product_name=RunnablePassthrough() 
)

# 6. SEQUENTIAL STEP: Combine the parallel results into a final plan
strategy_prompt = ChatPromptTemplate.from_template(
    """You are a CEO. Review the marketing and tech data to create a launch plan for {product_name}.
    
    Marketing Data: {marketing}
    Technical Data: {tech}
    
    {format_instructions}"""
).partial(format_instructions=final_parser.get_format_instructions())

# 7. ASSEMBLE THE COMPLETE ARCHITECTURE
# This is a Sequential Chain where the first step is actually two Parallel chains
full_system = parallel_brainstorm | strategy_prompt | model | final_parser

# 8. TEST IT
print("--- Generating Launch Strategy (Parallel + Sequential) ---")
result = full_system.invoke({"product_name": "Saas"})

print(f"PRODUCT: {result.product_name}")
print(f"STRATEGY: {textwrap.fill(result.strategy, width=80)}\n")
print(f"MARKETING SNIPPET: {textwrap.fill(result.marketing_summary, width=80)}\n")