from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# 1. Setup Model
model = ChatOllama(model="mistral-large-3:675b-cloud", temperature=0.2)

# 2. Define Chain A: Culture Specialist
culture_prompt = ChatPromptTemplate.from_template("What is a unique cultural fact about {city}?")
culture_chain = culture_prompt | model | StrOutputParser()

# 3. Define Chain B: Weather Specialist
weather_prompt = ChatPromptTemplate.from_template("Describe the typical weather in {city} during the summer.")
weather_chain = weather_prompt | model | StrOutputParser()

# 4. Define the Parallel Map
# This runs both chains simultaneously
map_chain = RunnableParallel(
    culture_info=culture_chain, 
    weather_info=weather_chain,
    city_name=RunnablePassthrough() # Passes the original input 'city' forward
)

# 5. Define the Final Combined Prompt
final_prompt = ChatPromptTemplate.from_template("""
Welcome to {city_name}!

CULTURE:
{culture_info}

WEATHER:
{weather_info}

Summary: Based on the above, tell me why I should visit in 1 sentence.
""")

# 6. Build the Full Chain
full_chain = map_chain | final_prompt | model | StrOutputParser()

# 7. Execute
print("--- Starting Parallel Processing ---")
result = full_chain.invoke({"city": "Tokyo"})

print(result)