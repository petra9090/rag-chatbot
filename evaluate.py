import os
import re
import json
import time
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
import anthropic
from dotenv import load_dotenv

load_dotenv()

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = Anthropic(model="claude-sonnet-4-6")

# ── Load index ──────────────────────────────────────────────────────────────
storage_context = StorageContext.from_defaults(persist_dir="storage")
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine(similarity_top_k=5)

client = anthropic.Anthropic()

# ── Helpers ──────────────────────────────────────────────────────────────────
def llm_judge(prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        except anthropic.APIStatusError as e:
            if e.status_code == 529 and attempt < retries - 1:
                wait = 10 * (attempt + 1)  # 10s, 20s, 30s
                print(f"  API overloaded, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

def parse_score(text: str, fallback: float = 0.5) -> float:
    match = re.search(r'(?:score|rating)[:\s]*([0-9]*\.?[0-9]+)', text.lower())
    if match:
        val = float(match.group(1))
        return val if val <= 1.0 else val / 100.0
    match = re.search(r'\b(0\.\d+|1\.0|0|1)\b', text)
    if match:
        return float(match.group(1))
    return fallback

# ── RAGAS Metrics ────────────────────────────────────────────────────────────
def faithfulness(answer: str, contexts: list[str]) -> float:
    context_block = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    claims_resp = llm_judge(
        f"Extract all atomic factual claims from this answer. "
        f"List each on a new line starting with '- '.\n\nAnswer: {answer}"
    )
    claims = [l.strip().lstrip("- ").strip()
              for l in claims_resp.split("\n") if l.strip().startswith("-")]
    if not claims:
        claims = [answer]

    supported = 0
    for claim in claims:
        verdict = llm_judge(
            f"Is this claim SUPPORTED or UNSUPPORTED by the context?\n"
            f"Reply with one word only: SUPPORTED or UNSUPPORTED.\n\n"
            f"Context:\n{context_block}\n\nClaim: {claim}"
        ).upper()
        if "SUPPORTED" in verdict and "UNSUPPORTED" not in verdict:
            supported += 1
    return supported / len(claims)

def answer_relevancy(question: str, answer: str) -> float:
    resp = llm_judge(
        f"Rate how relevant the answer is to the question (0.0 to 1.0).\n"
        f"Reply ONLY with JSON: {{\"score\": <float>}}\n\n"
        f"Question: {question}\nAnswer: {answer}"
    )
    return parse_score(resp)

def context_precision(question: str, contexts: list[str], ground_truth: str) -> float:
    verdicts = []
    for ctx in contexts:
        resp = llm_judge(
            f"Is this context RELEVANT or IRRELEVANT for answering the question "
            f"given the correct answer? Reply with one word only.\n\n"
            f"Question: {question}\nCorrect answer: {ground_truth}\nContext: {ctx}"
        ).upper()
        verdicts.append("RELEVANT" in resp and "IRRELEVANT" not in resp)

    relevant, precision_sum = 0, 0.0
    for i, v in enumerate(verdicts):
        if v:
            relevant += 1
            precision_sum += relevant / (i + 1)
    return precision_sum / max(relevant, 1)

def context_recall(contexts: list[str], ground_truth: str) -> float:
    context_block = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    sentences = [s.strip() for s in re.split(r'[.!?]+', ground_truth) if s.strip()]
    if not sentences:
        return 1.0

    attributed = 0
    for sent in sentences:
        resp = llm_judge(
            f"Can this statement be attributed to the contexts? Reply YES or NO only.\n\n"
            f"Contexts:\n{context_block}\n\nStatement: {sent}"
        ).upper()
        if "YES" in resp:
            attributed += 1
    return attributed / len(sentences)

# ── Evaluation dataset ────────────────────────────────────────────────────────
# Replace these with real questions and answers from YOUR course notes!
eval_questions = [
    {
        "question": "What is the difference between observational and experimental research design?",
        "ground_truth": "In observational studies researchers simply observe and measure variables without actively intervening. In experimental research, variables are purposefully manipulated to determine a cause-and-effect relationship.",
    },
    {
        "question": "What are the four main dimensions of data quality?",
        "ground_truth": "The four main dimensions of data quality are correctness/accuracy, completeness, conformity/validity, and consistency.",
    },
    {
        "question": "What is the difference between classical and Bayesian statistics?",
        "ground_truth": "Classical statistics is a set of tools for decision making using hypothesis testing. Bayesian statistics interprets probability as a degree of belief in an event, which can be updated as new evidence is obtained.",
    },
    {
        "question": "What is a p-value and what does a low p-value mean?",
        "ground_truth": "The p-value quantifies the probability of observing the value of the test statistic, or a more extreme value, under the null hypothesis. Low p-values are coherent with a rejection of the null hypothesis stating that there is no effect.",
    },
    {
        "question": "What is the purpose of blinding in experiments and what are the types?",
        "ground_truth": "Blinding eliminates conscious and unconscious influences on the treatment result. Types include open (no blinding), single-blind (participants unaware), double-blind (participants and researchers unaware), and triple-blind (participants, researchers, and analysts unaware).",
    },
    {
        "question": "What is web scraping?",
        "ground_truth": "Web scraping is data scraping used for extracting data from websites. It is a form of copying in which specific data is gathered and copied from the web, typically into a central local database or spreadsheet for later retrieval or analysis.",
    },
    {
        "question": "What is the purpose of virtual environments in Python?",
        "ground_truth": "Virtual environments are used to avoid dependency issues of packages and keep the development environment clean.",
    },
    {
        "question": "What is R and what is it used for?",
        "ground_truth": "R is a powerful programming language and software environment specifically designed for statistical computing, data analysis, and high-quality graphical visualization.",
    },
    {
        "question": "When should log-transformations be applied to a time series?",
        "ground_truth": "Log-transformations should be applied when the distribution of the time series elements is right-skewed, or when a time series could be produced by the product of model outcomes.",
    },
    {
        "question": "What is a confounder in research design?",
        "ground_truth": "A confounder is a factor that has not been investigated but is associated with both the independent and dependent variables, causing a spurious correlation between them.",
    },
    {
        "question": "What is R-squared and what is it used for in regression?",
        "ground_truth": "R-squared quantifies the grade of fit of a regression model and is calculated as the squared correlation between observed and predicted values. It is not used to formally compare models.",
    },
    {
        "question": "What are Generalised Linear Models and when are they used?",
        "ground_truth": "Generalised Linear Models (GLMs) generalise the linear model such that non-normal data can be analysed. They are used when the response variable represents count data (Poisson model) or binary/binomial data (Binomial model).",
    },
    {
        "question": "What is the logit link function and why is it used for binomial data?",
        "ground_truth": "The logit function is log(y / (1-y)) for y in [0,1]. It is used as a link function for binary or binomial data to constrain predictions within the [0%, 100%] range.",
    },
    {
        "question": "What are the assumptions of a linear model about the errors?",
        "ground_truth": "The errors must follow a normal distribution, have an expected value of zero, be homoscedastic (constant variance), and be independent.",
    },
]

# ── Run evaluation ────────────────────────────────────────────────────────────
print("Running RAGAS evaluation...\n")
results = []

for i, item in enumerate(eval_questions):
    time.sleep(3)  # wait 3 seconds between questions
    print(f"Evaluating Q{i+1}: {item['question'][:60]}...")

    # Get RAG response
    response = query_engine.query(item["question"])
    answer = str(response)
    contexts = [n.get_content() for n in response.source_nodes]

    # Score
    f_score  = faithfulness(answer, contexts)
    ar_score = answer_relevancy(item["question"], answer)
    cp_score = context_precision(item["question"], contexts, item["ground_truth"])
    cr_score = context_recall(contexts, item["ground_truth"])

    results.append({
        "question": item["question"],
        "answer": answer,
        "faithfulness": f_score,
        "answer_relevancy": ar_score,
        "context_precision": cp_score,
        "context_recall": cr_score,
    })

    print(f"  Faithfulness:      {f_score:.3f}")
    print(f"  Answer Relevancy:  {ar_score:.3f}")
    print(f"  Context Precision: {cp_score:.3f}")
    print(f"  Context Recall:    {cr_score:.3f}\n")

# ── Summary ───────────────────────────────────────────────────────────────────
means = {m: sum(r[m] for r in results) / len(results)
         for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]}

overall = 4 / sum(1.0 / max(v, 0.001) for v in means.values())

print("=" * 50)
print("RAGAS Summary")
print("=" * 50)
for metric, score in means.items():
    print(f"  {metric:<22} {score:.3f}")
print(f"\n  Overall (harmonic mean): {overall:.3f}")

# Save results to JSON
with open("ragas_results.json", "w") as f:
    json.dump({"summary": means, "overall": overall, "samples": results}, f, indent=2)
print("\nResults saved to ragas_results.json")