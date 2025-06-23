from ragas.metrics import LLMContextPrecisionWithoutReference
from ragas.metrics import Faithfulness
# from langchain_google_vertexai import ChatVertexAI
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.metrics import LLMContextPrecisionWithReference
from ragas.llms import LangchainLLMWrapper
from ragas import SingleTurnSample
from ragas.metrics import LLMContextRecall


evaluator_llm = LangchainLLMWrapper(
    ChatGoogleGenerativeAI(
                 model="gemini-2.0-flash", 
                 temperature=0,
                 google_api_key="your-gemini-api-key",
                 top_k = 1,
                 top_p=None,
                 )
)

def compute_faithfulness(query: str, response: str, retrieved_contexts: list[str]) -> float:
    sample = SingleTurnSample(
        user_input=query,
        response=response,
        retrieved_contexts=retrieved_contexts
    )
    scorer = Faithfulness(llm=evaluator_llm)
    return float(scorer.single_turn_score(sample))





def compute_context_precision_with_reference_llm(retrieved_contexts: list[str], query: str, response) -> float:

    sample = SingleTurnSample(
        user_input=query,
        reference=response,
        retrieved_contexts=retrieved_contexts, 
    )
    scorer = LLMContextPrecisionWithReference(llm=evaluator_llm)
    return float(scorer.single_turn_score(sample))

def compute_context_precision_without_reference(query: str, response: str, retrieved_contexts: list[str]) -> float:
    sample = SingleTurnSample(
        user_input=query,
        response=response,
        retrieved_contexts=retrieved_contexts
    )
    scorer = LLMContextPrecisionWithoutReference(llm=evaluator_llm)
    return float(scorer.single_turn_score(sample))


def compute_context_recall_with(query, reponse, retrieved_contexts: list[str]) -> float:
    sample = SingleTurnSample(
        user_input=query,
        response=reponse,
        reference=reponse,
        retrieved_contexts=retrieved_contexts, 
    )
    scorer = LLMContextRecall(llm=evaluator_llm)
    return float(scorer.single_turn_score(sample))
    

def weighted_average(metrics: dict[str, float], weights: dict[str, float]) -> float:
    return sum(metrics[key] * weights.get(key, 0) for key in metrics)

def arithmetic_mean(metrics: dict[str, float]) -> float:
    return sum(metrics.values()) / len(metrics)

def harmonic_mean(metrics: dict[str, float]) -> float:
    values = list(metrics.values())
    if all(v > 0 for v in values):
        return len(values) / sum(1 / v for v in values)
    return 0.0
    
    
    
    
def evaluate_configuration(
    query: str,
    response: str,
    retrieved_contexts: list[dict],
    reference_contexts: list[str],
    chunk_method: str = None,
    embedding_model: str = None,
    document_id: str = None
    
) -> dict:
    if not isinstance(reference_contexts, list):
        reference_contexts = [reference_contexts]
    
    
    present_reference_document = any(str(str(document_id) in element["document_id"]) for element in retrieved_contexts)
    
    
    retrieved_contexts = [context["content"] for context in retrieved_contexts]

    metrics = {
        "faithfulness": compute_faithfulness(
            query=query,
            response=response,
            retrieved_contexts=retrieved_contexts),
        "context_precision_with_reference": compute_context_precision_with_reference_llm(
            response=response,
            query=query,
            retrieved_contexts=retrieved_contexts
            ),
        "context_precision_without_reference": compute_context_precision_without_reference(
            query=query,
            response=response,
            retrieved_contexts=retrieved_contexts
            ),
        "context_recall": compute_context_recall_with(
            retrieved_contexts=retrieved_contexts,
            query=query,
            reponse=response,
            )
    }

    weights = {
        "faithfulness": 0.30,
        "context_precision_with_reference": 0.25,
        "context_precision_without_reference": 0.10,
        "context_recall": 0.35,
    }

    scores = {
        "weighted_average": weighted_average(metrics, weights),
        "harmonic_mean": harmonic_mean(metrics),
        "arithmetic_mean": arithmetic_mean(metrics),
    }
    
    
    return {
        "ragas": {
            "metrics": metrics,
            "scores": scores,
            },
    }
    