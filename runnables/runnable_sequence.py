import time
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableSequence

# ==========================================
# PART 1: DEFINING SIMPLE "WORKERS"
# ==========================================
# In LangChain, any function can be a step in the chain.
# We wrap them in RunnableLambda to make them compatible with the pipe (|).

def step1_clean_text(text: str):
    print(f"   [Step 1] Cleaning text: '{text}'")
    return text.strip().lower()

def step2_add_prefix(text: str):
    print(f"   [Step 2] Adding prefix to: '{text}'")
    return f"processed: {text}"

def step3_count_chars(text: str):
    print(f"   [Step 3] Counting chars in: '{text}'")
    return len(text)

# Turn them into Runnables
cleaner = RunnableLambda(step1_clean_text)
prefixer = RunnableLambda(step2_add_prefix)
counter = RunnableLambda(step3_count_chars)

# ==========================================
# PART 2: THE LINEAR SEQUENCE (A | B | C)
# ==========================================
print("\n--- EXPERIMENT 1: The Linear Chain ---")
print("Goal: Clean -> Prefix -> Count")

# The pipe symbol (|) creates a RunnableSequence automatically.
# Data flows: Input -> cleaner -> prefixer -> counter -> Output
linear_chain = cleaner | prefixer | counter

input_data = "   HELLO WORLD   "
print(f"INPUT: '{input_data}'")

result = linear_chain.invoke(input_data)
print(f"FINAL RESULT: {result}")


# ==========================================
# PART 3: THE PARALLEL SPLIT (The Dictionary)
# ==========================================
print("\n--- EXPERIMENT 2: The Branching Chain ---")
print("Goal: Use the input in two places at once (like RAG)")

# This mimics the {"context": ..., "question": ...} pattern from your RAG code.
branching_chain = (
    cleaner 
    | {
        "original_data": RunnablePassthrough(),  # Keeps the output of 'cleaner'
        "length_data": counter,                  # Calculates length of 'cleaner' output
        "custom_message": RunnableLambda(lambda x: f"We processed '{x}'") 
      }
)

# Flow:
# 1. "   HELLO WORLD   " -> cleaner -> "hello world"
# 2. "hello world" is BROADCAST to all 3 keys in the dictionary:
#    - original_data: receives "hello world" -> keeps "hello world"
#    - length_data:   receives "hello world" -> calculates 11
#    - custom_message: receives "hello world" -> makes string

print(f"INPUT: '{input_data}'")
result_dict = branching_chain.invoke(input_data)
print("FINAL RESULT (Dictionary):")
for key, value in result_dict.items():
    print(f"  - {key}: {value}")


# ==========================================
# PART 4: THE FULL "RAG-STYLE" SIMULATION
# ==========================================
print("\n--- EXPERIMENT 3: Simulating the RAG Pipeline ---")

# Let's simulate the components without using an actual LLM/Database
fake_retriever = RunnableLambda(lambda q: ["Doc A", "Doc B"]) # Pretends to search
fake_prompt = RunnableLambda(lambda x: f"PROMPT: Answer '{x['question']}' using {x['context']}")
fake_llm = RunnableLambda(lambda p: f"AI ANSWER based on: {p}")

# The classic RAG pattern
rag_simulation = (
    {
        "context": fake_retriever,
        "question": RunnablePassthrough()
    }
    | fake_prompt
    | fake_llm
)

query = "What is LangChain?"
print(f"Querying: {query}")
final_answer = rag_simulation.invoke(query)
print(f"Output: {final_answer}")