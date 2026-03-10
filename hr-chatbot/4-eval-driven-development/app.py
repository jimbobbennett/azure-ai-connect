from phoenix.otel import register

from dotenv import load_dotenv
from chat_bot import call_chat_bot

# Load environment variables from the .env file
load_dotenv()

# configure the Phoenix tracer
tracer_provider = register(
  auto_instrument=True # Auto-instrument your app based on installed dependencies
)

response = call_chat_bot("How many vacation days do I have left this year?", "jim")
print(response)
