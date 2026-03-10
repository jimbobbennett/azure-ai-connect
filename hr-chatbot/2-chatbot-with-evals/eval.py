import os

from phoenix.client import Client
from phoenix.evals import evaluate_dataframe, LLM
from phoenix.evals.metrics.faithfulness import FaithfulnessEvaluator
from phoenix.evals.utils import to_annotation_dataframe

from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Load the spans for the project into a DataFrame
spans_df = Client().spans.get_spans_dataframe(project_name=os.environ.get("PHOENIX_PROJECT_NAME"))

# The spans_df contains a row for each LLM call, along with the input and output messages in the attributes.
# We can transform this into a format suitable for evaluation by extracting the relevant information into new columns.
eval_df = spans_df[["context.span_id", "attributes.llm.input_messages", "attributes.llm.output_messages"]].copy()
eval_df["context"] = eval_df["attributes.llm.input_messages"].apply(lambda x: x[0]["message.content"])
eval_df["input"] = eval_df["attributes.llm.input_messages"].apply(lambda x: x[1]["message.content"])
eval_df["output"] = eval_df["attributes.llm.output_messages"].apply(lambda x: x[0]["message.content"])

# delete the original columns we no longer need
eval_df.drop(columns=["attributes.llm.input_messages", "attributes.llm.output_messages"], inplace=True)

# Add the index. This is needed to tie the spans back to the original spans in Phoenix after evaluation.
eval_df.set_index("context.span_id", inplace=True)

# Create an evaluator instance.
# Here we're using a faithfulness evaluator, which checks if the output of the LLM is faithful to the input context.
llm = LLM(provider="openai", model="gpt-4o-mini")
faithfulness_eval = FaithfulnessEvaluator(llm=llm)

# Evaluate the DataFrame using the evaluator. This will return a new DataFrame with the evaluation results.
results_df = evaluate_dataframe(dataframe=eval_df, evaluators=[faithfulness_eval])

# Convert the evaluation results into a format suitable for logging as span annotations in Phoenix, and log them.
annotation_df = to_annotation_dataframe(dataframe=results_df)
Client().spans.log_span_annotations_dataframe(dataframe=annotation_df)

# Write the latest annotation result to the console for easy viewing.
print(f"Score: {annotation_df.head(1)['score'].to_string(index=False)}")
print(f"Explanation: {annotation_df.head(1)['explanation'].values[0]}")
    