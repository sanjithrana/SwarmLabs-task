from evaluate import load
from langchain.llms import OpenAI

# ------------------- EVALUATION USING METRICS -------------------
def evaluate_rag_outputs(predictions, references):
    rouge = load("rouge")
    bleu = load("bleu")

    rouge_score = rouge.compute(predictions=predictions, references=references)
    bleu_score = bleu.compute(predictions=predictions, references=references)

    print("📊 ROUGE:", rouge_score)
    print("📘 BLEU:", bleu_score)

# ------------------- LLM AS A JUDGE -------------------
def llm_judge(prediction, reference):
    llm = OpenAI(model="gpt-3.5-turbo")
    prompt = f"""
    Evaluate the quality of the given answer compared to the reference.
    Reference: {reference}
    Answer: {prediction}
    Rate the accuracy on a scale of 1-10 and explain shortly.
    """
    judgment = llm.invoke(prompt)
    print("\n🤖 LLM Judgment:", judgment)

# ------------------- SAMPLE TEST -------------------
if __name__ == "__main__":
    predictions = [
        "Renewable energy reduces greenhouse gases and is sustainable long-term."
    ]
    references = [
        "Renewable energy helps reduce emissions and supports sustainable development."
    ]

    evaluate_rag_outputs(predictions, references)
    llm_judge(predictions[0], references[0])
