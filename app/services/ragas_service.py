import os
from typing import List, Dict, Any
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# Map string names to metric objects
AVAILABLE_METRICS = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall
}

class RagasService:
    @staticmethod
    def evaluate_response(
        question: str,
        answer: str,
        contexts: List[str],
        metrics: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate RAG response using Ragas metrics
        """
        # Select metrics to use
        selected_metrics = []
        for m in metrics:
            if m in AVAILABLE_METRICS:
                selected_metrics.append(AVAILABLE_METRICS[m])
        
        if not selected_metrics:
            raise ValueError("No valid metrics selected")

        # Prepare dataset
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            # If ground_truth is needed for other metrics, it would go here
            # "ground_truth": [ground_truth] 
        }
        dataset = Dataset.from_dict(data)

        #ToDO - нужно добавить возможность передавать llm
        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI
        evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        ragas_llm = LangchainLLMWrapper(evaluator_llm)
        # Evaluate
        # Note: Ragas uses OpenAI by default. Ensure OPENAI_API_KEY is set in environment.
        print(f"Selected metrics: {selected_metrics}")
        results = evaluate(
            dataset=dataset,
            metrics=selected_metrics,
            llm=ragas_llm
        )

        from ragas.evaluation import EvaluationResult
        if (isinstance(results, EvaluationResult)):
            print("printing ragas traces")
            print(results.ragas_traces)
            # print(results.scores) # [{'faithfulness': 1.0, 'answer_relevancy': np.float64(0.0)}]
            return results.scores[0]
        else:
            return {"error": 0.0}