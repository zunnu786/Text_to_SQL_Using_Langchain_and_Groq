from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
api_key = os.environ["GROQ_API_KEY"]

# Define the LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.6,
    max_retries=2,
    api_key=api_key,
    max_tokens=200,
)

def get_gemini_response(llm, question, prompt):
    # Ensure prompt is a string
    if isinstance(prompt, list):
        prompt = prompt[0]

    # Build prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", prompt),
        ("user", "{question}")
    ])

    # Format the messages for the LLM
    formatted_messages = prompt_template.format_messages(question=question)

    # Debug: print the messages being sent
    for msg in formatted_messages:
        print(f"{msg.type.upper()}: {msg.content}")

    # Get LLM response
    response = llm.invoke(formatted_messages)

    return response.content

# Define the prompt (as a string, not a list)
prompt = """
You are an expert in converting English questions to SQL query!
The SQL database has the name STUDENT and has the following columns - NAME, CLASS, SECTION

Example 1 - How many entries of records are present?,
the SQL command will be something like this:
SELECT COUNT(*) FROM STUDENT;

Example 2 - Tell me all the students studying in Data Science class?, 
the SQL command will be something like this:
SELECT * FROM STUDENT where CLASS="Data Science";

Also, the SQL code should not have ``` in beginning or end and do not include the word 'sql' in output.
"""

# Ask a question
result = get_gemini_response(llm, question="tell me all the students name", prompt=prompt)

print("\nGenerated SQL:\n", result)
