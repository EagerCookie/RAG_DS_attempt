from pydantic import BaseModel
from typing import List, Optional, Dict

class RagasEvalRequest(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    metrics: List[str] = ["faithfulness", "answer_relevancy"]
    ground_truth: Optional[str] = None

class RagasEvalResponse(BaseModel):
    scores: Dict[str, Optional[float]]
