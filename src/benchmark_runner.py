from transformers import pipeline
from metrics_logger import measure_inference_time, log_metrics

models = [
    "google/flan-t5-xsmall",
    "google/flan-t5-small",
    "google/flan-t5-base"
]

query = "What is the mean income for the state of Johor?"

for model_name in models:
    print(f"\nLoading model: {model_name}")
    generator = pipeline("text2text-generation", model=model_name, max_length=512)
    
    prompt = f"""Using the following context, answer the question.
    Context: Johor 1,161 1,465
    Question: {query}
    Answer:"""

    result, duration = measure_inference_time(generator, prompt, max_length=512, num_return_sequences=1)
    answer = result[0]['generated_text']

    print(f"Answer: {answer}")
    print(f"Time taken: {duration}s")

    log_metrics(model_name, query, answer, duration)
