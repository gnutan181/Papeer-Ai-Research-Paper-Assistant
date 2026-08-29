# import json
# import sys
# from pathlib import Path
# from uuid import uuid4

# sys.stdout.reconfigure(encoding="utf-8")
# sys.stderr.reconfigure(encoding="utf-8")
# from deepeval.models.base_model import DeepEvalBaseLLM
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# from langchain_core.messages import HumanMessage

# from deepeval import evaluate
# from deepeval.evaluate import AsyncConfig
# from deepeval.metrics import (
#     AnswerRelevancyMetric,
#     ContextualPrecisionMetric,
#     ContextualRecallMetric,
#     ContextualRelevancyMetric,
#     FaithfulnessMetric,
# )
# from deepeval.synthesizer import Synthesizer
# from deepeval.synthesizer.config import ContextConstructionConfig
# from deepeval.test_case import LLMTestCase

# from backend.paper_loader import load_document
# from backend.rag_graph import build_graph
# from backend.vector_store import add_paper

# load_dotenv()

# BASE_DIR = Path(__file__).resolve().parent
# PDF_PATH = BASE_DIR / "Openclaw_Research_Report.pdf"
# GOLDENS_FILE = BASE_DIR / "goldens.json"
# MAX_CONTEXTS        = 5
# GOLDENS_PER_CONTEXT = 2
# METRIC_THRESHOLD    = 0.7


# # def generate_goldens() -> list[dict]:
# #     synthesizer = Synthesizer()
# #     goldens = synthesizer.generate_goldens_from_docs(
# #         document_paths=[PDF_PATH],
# #         include_expected_output=True,
# #         max_goldens_per_context=GOLDENS_PER_CONTEXT,
# #         context_construction_config=ContextConstructionConfig(
# #             max_contexts_per_document=MAX_CONTEXTS,
# #         ),
# #     )
# #     print("goldens",goldens)
# #     pairs = [
# #         {"input": g.input, "expected_output": g.expected_output}
# #         for g in goldens
# #         if g.input and g.expected_output
# #     ]
# #     GOLDENS_FILE.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")
# #     return pairs
# class GroqDeepEvalModel(DeepEvalBaseLLM):

#     def __init__(
#         self,
#         model_name: str = "openai/gpt-oss-20b",
#     ):
#         self.model = ChatGroq(
#             model=model_name,
#             temperature=0,
#         )

#     def load_model(self):
#         return self.model

#     def generate(self, prompt: str, schema=None):

#         if schema is not None:
#             structured_model = self.model.with_structured_output(
#                 schema,
#                 method="json_schema",
#             )

#             return structured_model.invoke(prompt)

#         response = self.model.invoke(prompt)

#         return response.content

#     async def a_generate(self, prompt: str, schema=None):

#         if schema is not None:
#             structured_model = self.model.with_structured_output(
#                 schema,
#                 method="json_schema",
#             )

#             return await structured_model.ainvoke(prompt)

#         response = await self.model.ainvoke(prompt)

#         return response.content

#     def get_model_name(self):
#         return "Groq openai/gpt-oss-20b"
# def generate_goldens() -> list[dict]:
#     # Use your application's working PDF loader.
#     docs = load_document(str(PDF_PATH))

#     if not docs:
#         raise RuntimeError(f"No documents extracted from {PDF_PATH}")

#     # Convert LangChain Documents into DeepEval contexts.
#     # Each context is List[str].
#     contexts = [
#         [doc.page_content]
#         for doc in docs
#         if doc.page_content and doc.page_content.strip()
#     ]

#     if not contexts:
#         raise RuntimeError(
#             f"No usable text extracted from {PDF_PATH}"
#         )

#     print(f"Loaded {len(docs)} documents")
#     print(f"Created {len(contexts)} contexts for DeepEval")

#     # Limit the number of contexts if desired.
#     contexts = contexts[:MAX_CONTEXTS]

#     synthesizer = Synthesizer(
#     model=GroqDeepEvalModel(),
#     max_concurrent=1,
# )

#     goldens = synthesizer.generate_goldens_from_contexts(
#         contexts=contexts,
#         include_expected_output=True,
#         max_goldens_per_context=GOLDENS_PER_CONTEXT,
#     )

#     pairs = [
#         {
#             "input": g.input,
#             "expected_output": g.expected_output,
#         }
#         for g in goldens
#         if g.input and g.expected_output
#     ]

#     if not pairs:
#         raise RuntimeError(
#             "DeepEval generated 0 goldens from the extracted contexts."
#         )

#     GOLDENS_FILE.write_text(
#         json.dumps(
#             pairs,
#             indent=2,
#             ensure_ascii=False,
#         ),
#         encoding="utf-8",
#     )

#     print(f"Generated {len(pairs)} goldens")
#     return pairs

# def load_goldens() -> list[dict]:
#     return json.loads(GOLDENS_FILE.read_text(encoding="utf-8"))


# def run_rag_query(graph, query: str, session_id: str) -> tuple[str, list[str]]:
#     config = {"configurable": {"thread_id": str(session_id)}}
#     final_state = graph.invoke(
#         {
#             "messages": [HumanMessage(content=query)],
#             "session_id": session_id,
#             "query": query,
#             "retrieved_docs": [],
#             "retrieval_attempts": 0,
#             "rewrite_count": 0,
#         },
#         config=config,
#     )
#     answer = final_state.get("answer") or ""
#     retrieval_context = [doc.page_content for doc in (final_state.get("retrieved_docs") or [])]
#     return answer, retrieval_context


# # def main() -> None:
# #     pairs = load_goldens() if GOLDENS_FILE.exists() else generate_goldens()
# #     if not pairs:
# #         raise RuntimeError(
# #         "No goldens were generated. "
# #         "DeepEval could not construct contexts from the PDF."
# #         )

# #     print(f"Loaded {len(pairs)} golden test cases")
# #     # docs = load_document(PDF_PATH)
# #     docs = load_document(str(PDF_PATH))
# #     graph = build_graph(db_path="eval_checkpoints.db")

# #     metrics = [
# #         ContextualPrecisionMetric(threshold=METRIC_THRESHOLD, model="gpt-5.4-mini"),
# #         ContextualRecallMetric(threshold=METRIC_THRESHOLD, model="gpt-5.4-mini"),
# #         ContextualRelevancyMetric(threshold=METRIC_THRESHOLD, model="gpt-5.4-mini"),
# #         AnswerRelevancyMetric(threshold=METRIC_THRESHOLD, model="gpt-5.4-mini"),
# #         FaithfulnessMetric(threshold=METRIC_THRESHOLD, model="gpt-5.4-mini"),
# #     ]

# #     test_cases = []
# #     for pair in pairs:
# #         session_id = f"evaluation_session_{uuid4()}"
# #         add_paper(docs, session_id)

# #         query = pair["input"] + " as per the report in knowledge base"
# #         answer, retrieval_context = run_rag_query(graph, query, session_id)
# #         test_cases.append(
# #             LLMTestCase(
# #                 input=pair["input"],
# #                 actual_output=answer,
# #                 expected_output=pair["expected_output"],
# #                 retrieval_context=retrieval_context,
# #             )
# #         )

# #     results = evaluate(
# #         test_cases,
# #         metrics,
# #         async_config=AsyncConfig(max_concurrent=3, throttle_value=5),
# #     )

# #     summary = []
# #     for test_result in results.test_results:
# #         summary.append({
# #             "input": test_result.input,
# #             "actual_output": test_result.actual_output,
# #             "success": test_result.success,
# #             "metrics": [
# #                 {
# #                     "name": m.name,
# #                     "score": m.score,
# #                     "passed": m.success,
# #                     "reason": m.reason,
# #                 }
# #                 for m in test_result.metrics_data
# #             ],
# #         })

# #     results_path = Path("eval_results.json")
# #     results_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
# #     print(f"\nResults saved to {results_path}.")

# def main() -> None:
#     pairs = load_goldens() if GOLDENS_FILE.exists() else generate_goldens()

#     if not pairs:
#         raise RuntimeError(
#             "No goldens available for evaluation."
#         )

#     print(f"Loaded {len(pairs)} golden test cases")

#     docs = load_document(str(PDF_PATH))
#     graph = build_graph(db_path="eval_checkpoints.db")

#     metrics = [
#         ContextualPrecisionMetric(
#             threshold=METRIC_THRESHOLD,
#             model="gpt-5.4-mini",
#         ),
#         ContextualRecallMetric(
#             threshold=METRIC_THRESHOLD,
#             model="gpt-5.4-mini",
#         ),
#         ContextualRelevancyMetric(
#             threshold=METRIC_THRESHOLD,
#             model="gpt-5.4-mini",
#         ),
#         AnswerRelevancyMetric(
#             threshold=METRIC_THRESHOLD,
#             model="gpt-5.4-mini",
#         ),
#         FaithfulnessMetric(
#             threshold=METRIC_THRESHOLD,
#             model="gpt-5.4-mini",
#         ),
#     ]

#     test_cases = []

#     for pair in pairs:
#         session_id = f"evaluation_session_{uuid4()}"

#         add_paper(docs, session_id)

#         # Don't modify the synthetic question.
#         query = pair["input"]

#         answer, retrieval_context = run_rag_query(
#             graph,
#             query,
#             session_id,
#         )

#         test_cases.append(
#             LLMTestCase(
#                 input=pair["input"],
#                 actual_output=answer,
#                 expected_output=pair["expected_output"],
#                 retrieval_context=retrieval_context,
#             )
#         )

#     print(f"Created {len(test_cases)} evaluation test cases")

#     results = evaluate(
#         test_cases,
#         metrics,
#         async_config=AsyncConfig(
#             max_concurrent=3,
#             throttle_value=5,
#         ),
#     )

#     summary = []

#     for test_result in results.test_results:
#         summary.append(
#             {
#                 "input": test_result.input,
#                 "actual_output": test_result.actual_output,
#                 "success": test_result.success,
#                 "metrics": [
#                     {
#                         "name": m.name,
#                         "score": m.score,
#                         "passed": m.success,
#                         "reason": m.reason,
#                     }
#                     for m in test_result.metrics_data
#                 ],
#             }
#         )

#     results_path = BASE_DIR / "eval_results.json"

#     results_path.write_text(
#         json.dumps(
#             summary,
#             indent=2,
#             ensure_ascii=False,
#         ),
#         encoding="utf-8",
#     )

#     print(f"\nResults saved to {results_path}.")

# if __name__ == "__main__":
#     main()


import json
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

PDF_PATH = BASE_DIR / "Openclaw_Research_Report.pdf"

GOLDENS_FILE = BASE_DIR / "goldens.json"

RESULTS_FILE = BASE_DIR / "eval_results.json"

CHECKPOINT_DB = BASE_DIR / "eval_checkpoints.db"


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
GROQ_MODEL = "openai/gpt-oss-20b"

# Keep concurrency low for the free/on-demand TPM limit.
SYNTHESIZER_MAX_CONCURRENT = 1

EVALUATION_MAX_CONCURRENT = 1

THROTTLE_SECONDS = 10


# =========================================================
# Custom DeepEval model
# =========================================================

class GroqDeepEvalModel(DeepEvalBaseLLM):
    """
    Custom DeepEval LLM backed by Groq.

    Model:
        openai/gpt-oss-20b

    Important:
        DeepEval sometimes passes a Pydantic schema.
        In that case we MUST use structured output.

        Otherwise GPT-OSS may return plain text and
        schema.model_validate_json() will fail.
    """

    def __init__(
        self,
        model_name: str = GROQ_MODEL,
    ):
        self.model_name = model_name

        self.model = ChatGroq(
            model=model_name,
            temperature=0,
        )

    def load_model(self):
        return self.model

    def generate(
        self,
        prompt: str,
        schema=None,
    ):
        """
        Synchronous generation.

        If DeepEval provides a schema:
            use structured output.

        Otherwise:
            return normal text.
        """

        if schema is not None:
            structured_model = self.model.with_structured_output(
                schema,
                method="json_schema",
            )

            return structured_model.invoke(prompt)

        response = self.model.invoke(prompt)

        return response.content

    async def a_generate(
        self,
        prompt: str,
        schema=None,
    ):
        """
        Asynchronous generation.

        DeepEval's Synthesizer uses this method heavily.

        If schema is provided, use Groq structured output.
        """

        if schema is not None:
            structured_model = self.model.with_structured_output(
                schema,
                method="json_schema",
            )

            return await structured_model.ainvoke(prompt)

        response = await self.model.ainvoke(prompt)

        return response.content

    def get_model_name(self):
        return f"Groq {self.model_name}"


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

    deepeval_model = GroqDeepEvalModel(
        model_name=GROQ_MODEL
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

def load_goldens() -> list[dict]:
    """
    Load previously generated synthetic test cases.

    This prevents DeepEval from regenerating goldens
    every time evaluate.py is executed.
    """

    if not GOLDENS_FILE.exists():
        raise FileNotFoundError(
            f"Goldens file does not exist: {GOLDENS_FILE}"
        )

    pairs = json.loads(
        GOLDENS_FILE.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(pairs, list):
        raise RuntimeError(
            "goldens.json must contain a JSON list."
        )

    return pairs


# =========================================================
# Run RAG query
# =========================================================

def run_rag_query(
    graph,
    query: str,
    session_id: str,
) -> tuple[str, list[str]]:
    """
    Execute one query against the RAG graph.

    Returns:

        answer
        retrieval_context
    """

    config = {
        "configurable": {
            "thread_id": str(session_id)
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
        ContextualPrecisionMetric(
            threshold=METRIC_THRESHOLD,
            model=deepeval_model,
        ),

        ContextualRecallMetric(
            threshold=METRIC_THRESHOLD,
            model=deepeval_model,
        ),

        ContextualRelevancyMetric(
            threshold=METRIC_THRESHOLD,
            model=deepeval_model,
        ),

        AnswerRelevancyMetric(
            threshold=METRIC_THRESHOLD,
            model=deepeval_model,
        ),

        FaithfulnessMetric(
            threshold=METRIC_THRESHOLD,
            model=deepeval_model,
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
        # Unique RAG session
        # -------------------------------------------------

        session_id = (
            f"evaluation_session_{uuid4()}"
        )

        # -------------------------------------------------
        # Add paper to vector store
        # -------------------------------------------------

        add_paper(
            docs,
            session_id,
        )

        # -------------------------------------------------
        # Run RAG
        # -------------------------------------------------

        answer, retrieval_context = run_rag_query(
            graph=graph,
            query=query,
            session_id=session_id,
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
        f"\nEvaluation model: {GROQ_MODEL}"
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
        db_path=str(CHECKPOINT_DB)
    )

    print(
        "RAG graph ready"
    )

    # =====================================================
    # 4. Create evaluator model
    # =====================================================

    deepeval_model = GroqDeepEvalModel(
        model_name=GROQ_MODEL
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
            max_concurrent=EVALUATION_MAX_CONCURRENT,

            throttle_value=THROTTLE_SECONDS,
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