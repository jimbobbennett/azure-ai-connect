from phoenix.otel import register

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# configure the Phoenix tracer
tracer_provider = register(
  auto_instrument=True # Auto-instrument your app based on installed dependencies
)

# Initialize the OpenAI client
client = OpenAI()

# Define a system prompt with guidance
system_prompt = """
You are a helpful HR assistant. Provide very clear, very short,
and very succinct answers to the user's questions based off their
employment contract and other criteria.

Just provide the details asked, avoiding any extra information
or explanations.

When answering user questions, use the following context:
- The user's employment contract states that they have 30 days of annual leave.
- They have already taken 18 days of vacation this year.
"""

# Define a user prompt with a question
user_prompt = """
Question:

I am in the UK. How many vacation days do I have this year?
"""

# Send a request to the LLM
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
)

# Print the response
print(response.choices[0].message.content.strip())
