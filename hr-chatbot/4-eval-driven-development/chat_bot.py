import json
import uuid

from openinference.instrumentation import using_session
from openai import OpenAI


def get_vacation_days_for_user(user: str) -> dict[str, int]:
    """
    Get the number of vacation days for a given user,

    Return a dictionary with the total number of vacation days and the number of vacation days already taken.
    """
    # In a real application, this function would query a database or another data source to get the user's vacation days.
    # For this example, we'll just return a hardcoded value.
    return {"total_vacation_days": 30, "used_vacation_days": 18}


def call_chat_bot(question: str, user: str) -> str:
    # Initialize the OpenAI client
    client = OpenAI()

    # Define a system prompt with guidance
    system_prompt = f"""
    You are a helpful HR assistant. Provide very clear, very short,
    and very succinct answers to the user's questions based off their
    employment contract and other criteria.

    Just provide the details asked, avoiding any extra information
    or explanations.

    User is {user}
    """
    with using_session(str(uuid.uuid4())):
        # Define messages and tools
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_vacation_days_for_user",
                    "description": "Get the total vacation days and used vacation days for a user.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user": {
                                "type": "string",
                                "description": "The user identifier or name.",
                            }
                        },
                        "required": ["user"],
                    },
                },
            }
        ]

        # First LLM call (may return a tool call)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
        )

        message = response.choices[0].message
        if message.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                        for tool_call in message.tool_calls
                    ], #type: ignore
                }
            )

            for tool_call in message.tool_calls:
                if tool_call.function.name == "get_vacation_days_for_user":
                    args = json.loads(tool_call.function.arguments)
                    result = get_vacation_days_for_user(args["user"])

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result),
                        }
                    )

            # Second LLM call to produce final user-facing response
            final_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
            return final_response.choices[0].message.content.strip()

        # No tool call; return direct response
        return message.content.strip()
