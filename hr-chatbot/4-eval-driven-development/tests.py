from datetime import datetime
import time
import os

import pytest
from unittest.mock import patch
from dotenv import load_dotenv

from phoenix.client import Client
from phoenix.evals import evaluate_dataframe, LLM
from phoenix.evals.metrics.faithfulness import FaithfulnessEvaluator
from phoenix.otel import register

from chat_bot import call_chat_bot

load_dotenv()

PROMPT_CASES = [
    {
        "prompt": "How many vacation days do I get each year?",
        "user": "jim",
        "vacation": (20, 4),
    },
    {
        "prompt": "How many holiday days have I already taken?",
        "user": "alex",
        "vacation": (25, 7),
    },
    {
        "prompt": "How many PTO days do I still have left?",
        "user": "sam",
        "vacation": (30, 12),
    },
    {
        "prompt": "Can you tell me my total annual leave and what remains?",
        "user": "taylor",
        "vacation": (28, 3),
    },
    {
        "prompt": "I need my vacation summary: total, used, and left.",
        "user": "jordan",
        "vacation": (15, 10),
    },
    {
        "prompt": "What's my current vacation balance?",
        "user": "morgan",
        "vacation": (22, 8),
    },
    {
        "prompt": "Please check my leave entitlement for this year.",
        "user": "jamie",
        "vacation": (18, 2),
    },
    {
        "prompt": "Vacation days left?",
        "user": "drew",
        "vacation": (10, 1),
    },
    {
        "prompt": "How much PTO have I used so far in 2026?",
        "user": "casey",
        "vacation": (26, 14),
    },
    {
        "prompt": "Could you confirm how many annual leave days I'm entitled to?",
        "user": "riley",
        "vacation": (35, 5),
    },
    {
        "prompt": "Need a quick leave count: total allotment vs consumed days.",
        "user": "blair",
        "vacation": (24, 24),
    },
    {
        "prompt": "Based on my profile, what's my vacation allowance and remaining days?",
        "user": "quinn",
        "vacation": (16, 9),
    },
    {
        "prompt": "Give me my PTO totals in numbers only.",
        "user": "charlie",
        "vacation": (12, 11),
    },
    {
        "prompt": "I want to plan time off. How many days have I got left exactly?",
        "user": "sky",
        "vacation": (27, 13),
    },
    {
        "prompt": "Tell me annual leave taken and available.",
        "user": "reese",
        "vacation": (21, 6),
    },
    {
        "prompt": "Before I request leave, what are my total and used vacation days?",
        "user": "devon",
        "vacation": (29, 0),
    },
    {
        "prompt": "What's my PTO situation right now?",
        "user": "harper",
        "vacation": (14, 14),
    },
    {
        "prompt": "Please provide my vacation entitlement details for this leave year.",
        "user": "rowan",
        "vacation": (32, 17),
    },
    {
        "prompt": "How many paid leave days are still available to me?",
        "user": "emerson",
        "vacation": (19, 18),
    },
    {
        "prompt": "In one line: total leave, leave used, leave remaining.",
        "user": "finley",
        "vacation": (23, 15),
    },
]

# configure the Phoenix tracer
tracer_provider = register(
  auto_instrument=True # Auto-instrument your app based on installed dependencies
)

# Create an evaluator instance.
# Here we're using a faithfulness evaluator, which checks if the output of the LLM is faithful to the input context.
llm = LLM(provider="openai", model="gpt-4o-mini")
faithfulness_eval = FaithfulnessEvaluator(llm=llm)


@pytest.mark.parametrize("case", PROMPT_CASES)
def test_call_chat_bot(case: dict) -> None:
    # Arrange
    with patch(
        "chat_bot.get_vacation_days_for_user",
        return_value={"total_vacation_days": case["vacation"][0], "used_vacation_days": case["vacation"][1]},
    ):
        # Capture the current date time
        current_datetime = datetime.now()

        # Act
        response = call_chat_bot(case["prompt"], case["user"])
        print(response)

        # Sleep for 2 seconds to ensure the spans have been logged to Phoenix before we try to retrieve them for evaluation.
        time.sleep(2)

        # Evaluate

        # Load the spans for the project based off our start time
        spans_df = Client().spans.get_spans_dataframe(
            start_time=current_datetime,
            project_name=os.environ.get("PHOENIX_PROJECT_NAME")
        )

        # Build a dataframe for evaluation using the latest trace which will include all the messages.
        eval_df = spans_df[["context.span_id", "attributes.llm.input_messages", "attributes.llm.output_messages"]].head(1).copy()
        eval_df["input"] = eval_df["attributes.llm.input_messages"].apply(lambda x: x[1]["message.content"])
        eval_df["output"] = eval_df["attributes.llm.output_messages"].apply(lambda x: x[0]["message.content"])

        # Context comes from the tool call and the system prompt, so assemble these into one string for evaluation.
        eval_df["context"] = eval_df["attributes.llm.input_messages"].apply(lambda x: f"""{x[0]["message.content"]}
        Tool call input for get_vacation_days_for_user:
        {x[3]["message.content"]}
        """)

        # delete the original columns we no longer need
        eval_df.drop(columns=["attributes.llm.input_messages", "attributes.llm.output_messages"], inplace=True)

        # Add the index. This is needed to tie the spans back to the original spans in Phoenix after evaluation.
        eval_df.set_index("context.span_id", inplace=True)

        # Evaluate the DataFrame using the evaluator. This will return a new DataFrame with the evaluation results.
        results_df = evaluate_dataframe(dataframe=eval_df, evaluators=[faithfulness_eval])
        score = results_df["faithfulness_score"].head(1).values[0]["score"]
        explanation = results_df["faithfulness_score"].head(1).values[0]["explanation"]

        # Assert

        assert score == 1.0, f"Expected faithfulness score of 1.0 but got {score}. Explanation: {explanation}"
