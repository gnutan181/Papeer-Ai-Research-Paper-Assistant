

import json
import os
import sys
from pathlib import Path
from uuid import uuid4

# =========================================================
# Console encoding
# =========================================================

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


# =========================================================
# Environment
# =========================================================

from dotenv import load_dotenv

load_dotenv()

# DeepEval wraps an entire metric/test-case attempt in a deadline. Contextual
# relevancy can require several sequential judge calls, so its default 180-second
# budget is too small for Gemini on a standard/free service tier. Keep one
# DeepEval attempt and give that attempt a bounded five-minute allowance.
# These must be set before DeepEval is imported because its settings are cached.
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "600")
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "300")
os.environ.setdefault("DEEPEVAL_RETRY_MAX_ATTEMPTS", "1")


# =========================================================
# DeepEval
# =========================================================

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.models.base_model import DeepEvalBaseLLM

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)

from deepeval.synthesizer import Synthesizer
from deepeval.test_case import LLMTestCase


# =========================================================
# LangChain / Groq
# =========================================================

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


# =========================================================
# Project imports
# =========================================================

from backend.paper_loader import load_document
from backend.rag_graph import build_graph
from backend.vector_store import add_paper


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
print(f"Base directory: {BASE_DIR}")
PDF_PATH = BASE_DIR / "Openclaw_Research_Report.pdf"

GOLDENS_FILE = BASE_DIR / "goldens.json"

RESULTS_FILE = BASE_DIR / "eval_results.json"

CHECKPOINT_DB = BASE_DIR / "eval_checkpoints.db"
print(f"Checkpoint database: {CHECKPOINT_DB}")

# =========================================================
# Evaluation configuration
# =========================================================

# Start small because Groq currently gives you
# approximately 8,000 tokens/minute for this model.
MAX_CONTEXTS = 2

# Number of synthetic questions per context.
GOLDENS_PER_CONTEXT = 1

# DeepEval pass threshold.
METRIC_THRESHOLD = 0.7

# Your Groq model.
GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_MODEL = "gemini-3.1-flash-lite"
# Keep concurrency low for the free/on-demand TPM limit.
SYNTHESIZER_MAX_CONCURRENT = 1

EVALUATION_MAX_CONCURRENT = 1

THROTTLE_SECONDS = 10


# =========================================================
# Custom DeepEval model
# =========================================================

class GeminiDeepEvalModel(DeepEvalBaseLLM):
    """
    Custom DeepEval judge backed by Gemini native structured output.
    """

    def __init__(
        self,
        model_name: str = GEMINI_MODEL,
    ):
        self.model_name = model_name

        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            # Contextual relevancy is classification, not deep reasoning.
            thinking_level="minimal",
            max_tokens=2048,
            timeout=120,
            max_retries=1,
        )

    def load_model(self):
        return self.model

    # def generate(
    #     self,
    #     prompt: str,
    #     schema=None,
    # ):
    #     """
    #     Synchronous generation.

    #     If DeepEval provides a schema:
    #         use structured output.

    #     Otherwise:
    #         return normal text.
    #     """

    #     if schema is not None:
    #         structured_model = self.model.with_structured_output(
    #             schema,
    #             method="json_schema",
    #         )

    #         return structured_model.invoke(prompt)

    #     response = self.model.invoke(prompt)

    #     return response.content

    # async def a_generate(
    #     self,
    #     prompt: str,
    #     schema=None,
    # ):
    #     """
    #     Asynchronous generation.

    #     DeepEval's Synthesizer uses this method heavily.

    #     If schema is provided, use Groq structured output.
    #     """

    #     if schema is not None:
    #         structured_model = self.model.with_structured_output(
    #             schema,
    #             method="json_schema",
    #         )

    #         return await structured_model.ainvoke(prompt)

    #     response = await self.model.ainvoke(prompt)

    #     return response.content
    def generate(
    self,
    prompt: str,
    schema=None,
):
        if schema is not None:
               structured_model = self.model.with_structured_output(
                   schema,
                   method="json_schema",
                #    strict=True,
               )
       
               return structured_model.invoke(prompt)
       
        response = self.model.invoke(prompt)
        return response.content   


    async def a_generate(
    self,
    prompt: str,
    schema=None,
):
     if schema is not None:
        structured_model = self.model.with_structured_output(
            schema,
            method="json_schema",
            # strict=True,
        )

        return await structured_model.ainvoke(prompt)

     response = await self.model.ainvoke(prompt)
     return response.content
    def get_model_name(self):
        return f"Gemini {self.model_name}"


# =========================================================
# Generate synthetic golden test cases
# =========================================================

def generate_goldens() -> list[dict]:
    """
    Generate synthetic evaluation questions from the PDF.

    Pipeline:

        PDF
         ↓
        load_document()
         ↓
        LangChain Documents
         ↓
        DeepEval contexts
         ↓
        GPT-OSS-20B
         ↓
        goldens.json
    """

    print("\n" + "=" * 70)
    print("GENERATING GOLDENS")
    print("=" * 70)

    # -----------------------------------------------------
    # Load PDF
    # -----------------------------------------------------

    docs = load_document(str(PDF_PATH))

    if not docs:
        raise RuntimeError(
            f"No documents extracted from: {PDF_PATH}"
        )

    print(f"Loaded {len(docs)} documents")

    # -----------------------------------------------------
    # Convert documents to DeepEval contexts
    # -----------------------------------------------------

    contexts = [
        [doc.page_content]
        for doc in docs
        if doc.page_content
        and doc.page_content.strip()
    ]

    if not contexts:
        raise RuntimeError(
            f"No usable text extracted from: {PDF_PATH}"
        )

    print(
        f"Created {len(contexts)} contexts for DeepEval"
    )

    # -----------------------------------------------------
    # Limit contexts
    # -----------------------------------------------------

    contexts = contexts[:MAX_CONTEXTS]

    print(
        f"Using {len(contexts)} contexts for golden generation"
    )

    print(
        f"Goldens per context: {GOLDENS_PER_CONTEXT}"
    )

    print(
        f"Expected maximum goldens: "
        f"{len(contexts) * GOLDENS_PER_CONTEXT}"
    )

    # -----------------------------------------------------
    # Create DeepEval model
    # -----------------------------------------------------

    deepeval_model = GeminiDeepEvalModel(
        model_name=GEMINI_MODEL
    )

    # -----------------------------------------------------
    # DeepEval Synthesizer
    # -----------------------------------------------------

    synthesizer = Synthesizer(
        model=deepeval_model,
        max_concurrent=SYNTHESIZER_MAX_CONCURRENT,
    )

    # -----------------------------------------------------
    # Generate goldens
    # -----------------------------------------------------

    goldens = synthesizer.generate_goldens_from_contexts(
        contexts=contexts,
        include_expected_output=True,
        max_goldens_per_context=GOLDENS_PER_CONTEXT,
    )

    # -----------------------------------------------------
    # Convert DeepEval Golden objects
    # -----------------------------------------------------

    pairs = [
        {
            "input": golden.input,
            "expected_output": golden.expected_output,
        }
        for golden in goldens
        if golden.input
        and golden.expected_output
    ]

    if not pairs:
        raise RuntimeError(
            "DeepEval generated 0 goldens."
        )

    # -----------------------------------------------------
    # Save goldens
    # -----------------------------------------------------

    GOLDENS_FILE.write_text(
        json.dumps(
            pairs,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"\nGenerated {len(pairs)} goldens"
    )

    print(
        f"Saved goldens to: {GOLDENS_FILE}"
    )

    return pairs


# =========================================================
# Load existing goldens
# =========================================================

# def load_goldens() -> list[dict]:
#     """
#     Load previously generated synthetic test cases.

#     This prevents DeepEval from regenerating goldens
#     every time evaluate.py is executed.
#     """

#     if not GOLDENS_FILE.exists():
#         raise FileNotFoundError(
#             f"Goldens file does not exist: {GOLDENS_FILE}"
#         )

#     pairs = json.loads(
#         GOLDENS_FILE.read_text(
#             encoding="utf-8"
#         )
#     )

#     if not isinstance(pairs, list):
#         raise RuntimeError(
#             "goldens.json must contain a JSON list."
#         )

#     return pairs
def load_goldens() -> list[dict] | None:
    """
    Load existing goldens.

    Return None when the file is missing, empty,
    malformed, or contains no usable test cases.
    """

    if not GOLDENS_FILE.exists():
        return None

    raw_content = GOLDENS_FILE.read_text(
        encoding="utf-8-sig"
    ).strip()

    if not raw_content:
        print(
            f"Warning: {GOLDENS_FILE.name} is empty. "
            "Goldens will be regenerated."
        )
        return None

    try:
        pairs = json.loads(raw_content)
    except json.JSONDecodeError as error:
        print(
            f"Warning: {GOLDENS_FILE.name} contains invalid JSON: "
            f"{error}. Goldens will be regenerated."
        )
        return None

    if not isinstance(pairs, list) or not pairs:
        print(
            f"Warning: {GOLDENS_FILE.name} does not contain "
            "a non-empty JSON list. Goldens will be regenerated."
        )
        return None

    valid_pairs = [
        pair
        for pair in pairs
        if isinstance(pair, dict)
        and pair.get("input")
        and pair.get("expected_output")
    ]

    if not valid_pairs:
        print(
            f"Warning: No valid test cases found in "
            f"{GOLDENS_FILE.name}. Goldens will be regenerated."
        )
        return None

    return valid_pairs

# =========================================================
# Run RAG query
# =========================================================

def run_rag_query(
    graph,
    query: str,
    session_id: str,
    thread_id: str | None = None,
) -> tuple[str, list[str]]:
    """
    Execute one query against the RAG graph.

    Returns:

        answer
        retrieval_context
    """

    config = {
        "configurable": {
            "thread_id": str(thread_id or session_id)
        }
    }

    final_state = graph.invoke(
        {
            "messages": [
                HumanMessage(content=query)
            ],

            "session_id": session_id,

            "query": query,

            "retrieved_docs": [],

            "retrieval_attempts": 0,

            "rewrite_count": 0,
        },
        config=config,
    )

    # -----------------------------------------------------
    # Extract final answer
    # -----------------------------------------------------

    answer = (
        final_state.get("answer")
        or ""
    )

    # -----------------------------------------------------
    # Extract retrieved documents
    # -----------------------------------------------------

    retrieved_docs = (
        final_state.get("retrieved_docs")
        or []
    )

    retrieval_context = [
        doc.page_content
        for doc in retrieved_docs
        if getattr(doc, "page_content", None)
    ]

    return answer, retrieval_context


# =========================================================
# Create DeepEval metrics
# =========================================================

def create_metrics(
    deepeval_model: DeepEvalBaseLLM,
):
    """
    Create all DeepEval metrics using the same
    custom Groq evaluator.
    """

    return [
        # ContextualPrecisionMetric(
        #     threshold=METRIC_THRESHOLD,
        #     model=deepeval_model,
        #     include_reason=False,
        #     async_mode=False,
        # ),

        # ContextualRecallMetric(
        #     threshold=METRIC_THRESHOLD,
        #     model=deepeval_model,
        #     include_reason=False,
        #     async_mode=False,
        # ),

        # ContextualRelevancyMetric(
        #     threshold=METRIC_THRESHOLD,
        #     model=deepeval_model,
        #     include_reason=False,
        #     async_mode=False,
        # ),

        AnswerRelevancyMetric(
            threshold=METRIC_THRESHOLD,
            model=deepeval_model,
            include_reason=False,
            async_mode=False,
        ),

        FaithfulnessMetric(
            threshold=METRIC_THRESHOLD,
            model=deepeval_model,
            include_reason=False,
            async_mode=False,
        ),
    ]


# =========================================================
# Build evaluation test cases
# =========================================================

def build_test_cases(
    pairs: list[dict],
    docs,
    graph,
) -> list[LLMTestCase]:
    """
    Convert golden test cases into DeepEval LLMTestCase objects.
    """

    test_cases = []

    print("\n" + "=" * 70)
    print("RUNNING RAG TEST CASES")
    print("=" * 70)

    # All goldens query the same PDF corpus. Upload it once per evaluation run.
    # LangGraph thread IDs remain unique below, so test state cannot leak between
    # cases even though they share the same Qdrant session partition.
    corpus_session_id = f"evaluation_corpus_{uuid4()}"
    print(f"\nIndexing {len(docs)} document chunks once...")
    add_paper(
        docs,
        corpus_session_id,
        document_id=f"evaluation_{corpus_session_id}",
        document_title=PDF_PATH.stem,
    )
    print("Evaluation corpus indexed")

    for index, pair in enumerate(
        pairs,
        start=1,
    ):
        print(
            f"\nTest case {index}/{len(pairs)}"
        )

        # -------------------------------------------------
        # Validate golden
        # -------------------------------------------------

        query = pair.get("input")

        expected_output = pair.get(
            "expected_output"
        )

        if not query:
            print(
                "Skipping test case: empty input"
            )
            continue

        if not expected_output:
            print(
                "Skipping test case: empty expected_output"
            )
            continue

        # -------------------------------------------------
        # Run RAG
        # -------------------------------------------------

        answer, retrieval_context = run_rag_query(
            graph=graph,
            query=query,
            session_id=corpus_session_id,
            thread_id=f"evaluation_case_{index}_{uuid4()}",
        )

        print(
            f"Query: {query}"
        )

        print(
            f"Retrieved contexts: "
            f"{len(retrieval_context)}"
        )

        # -------------------------------------------------
        # Create DeepEval test case
        # -------------------------------------------------

        test_case = LLMTestCase(
            input=query,

            actual_output=answer,

            expected_output=expected_output,

            retrieval_context=retrieval_context,
        )

        test_cases.append(test_case)

    return test_cases


# =========================================================
# Save evaluation results
# =========================================================

def save_results(results):
    """
    Convert DeepEval results into JSON-friendly data.
    """

    summary = []

    for test_result in results.test_results:

        metric_results = []

        for metric in test_result.metrics_data:

            metric_results.append(
                {
                    "name": metric.name,

                    "score": metric.score,

                    "passed": metric.success,

                    "reason": metric.reason,
                }
            )

        summary.append(
            {
                "input": test_result.input,

                "actual_output": (
                    test_result.actual_output
                ),

                "success": test_result.success,

                "metrics": metric_results,
            }
        )

    RESULTS_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return summary


# =========================================================
# Print evaluation summary
# =========================================================

def print_summary(summary):
    """
    Print a readable evaluation summary.
    """

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    for index, result in enumerate(
        summary,
        start=1,
    ):

        print(
            f"\nTest Case {index}"
        )

        print(
            f"Input: {result['input']}"
        )

        print(
            f"Overall Passed: "
            f"{result['success']}"
        )

        print(
            "\nMetrics:"
        )

        for metric in result["metrics"]:

            score = metric["score"]

            if score is None:
                score_text = "N/A"
            else:
                score_text = f"{score:.4f}"

            print(
                f"  {metric['name']}: "
                f"{score_text} "
                f"| passed={metric['passed']}"
            )

            if metric["reason"]:
                print(
                    f"    Reason: "
                    f"{metric['reason']}"
                )


# =========================================================
# Main
# =========================================================

def main() -> None:

    print("\n" + "=" * 70)
    print("PAPEER AI - RAG EVALUATION")
    print("=" * 70)

    print(
        f"\nEvaluation model: {GEMINI_MODEL}"
    )

    print(
        f"Max contexts: {MAX_CONTEXTS}"
    )

    print(
        f"Goldens per context: "
        f"{GOLDENS_PER_CONTEXT}"
    )

    print(
        f"Synthesizer concurrency: "
        f"{SYNTHESIZER_MAX_CONCURRENT}"
    )

    print(
        f"Evaluation concurrency: "
        f"{EVALUATION_MAX_CONCURRENT}"
    )

    # =====================================================
    # 1. Load or generate goldens
    # =====================================================

    if GOLDENS_FILE.exists():

        print(
            f"\nExisting goldens found:"
        )

        print(
            GOLDENS_FILE
        )

        pairs = load_goldens()
        if pairs is None:
            print("\nNo valid goldens found.")
            print("Generating synthetic goldens...")
            pairs = generate_goldens()
        else:
            print(f"\nLoaded {len(pairs)} existing goldens")
            print(
                        f"Loaded {len(pairs)} existing goldens"
                    )

    else:

        print(
            "\nNo goldens.json found."
        )

        print(
            "Generating synthetic goldens..."
        )

        pairs = generate_goldens()

    if not pairs:

        raise RuntimeError(
            "No goldens available for evaluation."
        )

    # =====================================================
    # 2. Load PDF
    # =====================================================

    print("\nLoading PDF...")

    docs = load_document(
        str(PDF_PATH)
    )

    if not docs:

        raise RuntimeError(
            f"Could not load PDF: {PDF_PATH}"
        )

    print(
        f"Loaded {len(docs)} documents"
    )

    # =====================================================
    # 3. Build RAG graph
    # =====================================================

    print("\nBuilding RAG graph...")

    graph = build_graph(
        
    )

    print(
        "RAG graph ready"
    )

    # =====================================================
    # 4. Create evaluator model
    # =====================================================

    deepeval_model = GeminiDeepEvalModel(
        model_name=GEMINI_MODEL
    )

    # =====================================================
    # 5. Create metrics
    # =====================================================

    metrics = create_metrics(
        deepeval_model
    )

    print(
        "\nMetrics configured:"
    )

    for metric in metrics:
        print(
            f"  - {metric.__class__.__name__}"
        )

    # =====================================================
    # 6. Build test cases
    # =====================================================

    test_cases = build_test_cases(
        pairs=pairs,
        docs=docs,
        graph=graph,
    )

    if not test_cases:

        raise RuntimeError(
            "No valid DeepEval test cases were created."
        )

    print(
        f"\nCreated {len(test_cases)} evaluation "
        f"test cases"
    )

    # =====================================================
    # 7. Run DeepEval
    # =====================================================

    print("\n" + "=" * 70)
    print("RUNNING DEEPEVAL")
    print("=" * 70)

    results = evaluate(
        test_cases,

        metrics,

        async_config=AsyncConfig(
            # max_concurrent=EVALUATION_MAX_CONCURRENT,

            # throttle_value=THROTTLE_SECONDS,
            run_async=False
        ),
    )

    # =====================================================
    # 8. Save results
    # =====================================================

    summary = save_results(
        results
    )

    # =====================================================
    # 9. Print summary
    # =====================================================

    print_summary(
        summary
    )

    print("\n" + "=" * 70)

    print(
        f"Results saved to:"
    )

    print(
        RESULTS_FILE
    )

    print("=" * 70)


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    main()

