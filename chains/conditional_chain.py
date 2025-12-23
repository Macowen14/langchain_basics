from typing import List, Literal
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field
import textwrap

# --- 1. SET UP THE MODEL ---
# Using json format ensures our Pydantic parsers work reliably
model = ChatOllama(model="mistral-large-3:675b-cloud", temperature=0.1, format="json")

# --- 2. DEFINE STRUCTURED DATA (PYDANTIC) ---
class RouterDecision(BaseModel):
    """Decides the 'vibe' of the content based on the topic."""
    logic_path: Literal["educational", "persuasive"] = Field(
        description="Educational for science/facts, Persuasive for debates/opinions."
    )
    reasoning: str = Field(description="Briefly why you chose this path.")


class FinalArticle(BaseModel):
    """The final polished output."""
    title: str = Field(description="A compelling headline.")
    body: str = Field(description="The main content (at least 3 paragraphs).")
    word_count: int = Field(description="The approximate word count.")


# --- 3. INITIALIZE PARSERS ---
router_parser = PydanticOutputParser(pydantic_object=RouterDecision)
final_parser = PydanticOutputParser(pydantic_object=FinalArticle)



# --- 4. DEFINE PROMPTS ---

# Prompt 1: The Router (The "Brain")
router_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a content strategist. Categorize the topic.\n{format_instructions}"),
    ("human", "Topic: {topic}")
]).partial(format_instructions=router_parser.get_format_instructions())

# Prompt 2a: Educational Path
edu_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professor. Write a factual, neutral explanation about {topic}.\n{format_instructions}"),
]).partial(format_instructions=final_parser.get_format_instructions())

# Prompt 2b: Persuasive Path
persuade_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an influencer. Write a strong, opinionated argument about {topic}.\n{format_instructions}"),
]).partial(format_instructions=final_parser.get_format_instructions())

# --- 5. LOGIC AND CHAINING ---

def routing_logic(input_dict):
    """
    This function acts as the 'Fork in the Road'.
    It runs the router, looks at the decision, and returns the next chain to run.
    """
    # Run the router
    decision = router_chain.invoke({"topic": input_dict["topic"]})
    print(f"DEBUG: Decision made -> {decision.logic_path}")
    print(textwrap.fill(decision.reasoning, width=80))
    
    # Choose the next chain
    if decision.logic_path == "educational":
        return edu_prompt | model | final_parser
    else:
        return persuade_prompt | model | final_parser

# --- 6. ASSEMBLE THE FINAL CHAIN ---

# The router chain
router_chain = router_prompt | model | router_parser

# The full execution flow
# We use RunnableLambda to 'wrap' our python logic function
full_chain = RunnableLambda(routing_logic)

# --- 7. RUN THE SYSTEM ---

def run_content_bot(user_topic: str):
    print(f"\n--- Processing Topic: {user_topic} ---")
    result = full_chain.invoke({"topic": user_topic})
    
    print(f"\nTITLE: {result.title}")
    print("BODY:")
    print(textwrap.fill(result.body, width=80))
    print(f"WORDS: {result.word_count}")
    print("-" * 30)

if __name__ == "__main__":
    # Test Path 1: Should be Educational
    run_content_bot("How Photosynthesis Works")
    
    # Test Path 2: Should be Persuasive
    run_content_bot("Why Pizza is better than Tacos")