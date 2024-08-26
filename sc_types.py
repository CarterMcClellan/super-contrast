from typing import Dict, List, Optional, Union
from pydantic import BaseModel


class ComparisonItem(BaseModel):
    path: Optional[str]
    data: Optional[str]
    true_label: str
    model_pred: str


class GroupedComparison(BaseModel):
    model_preds: Dict[str, List[ComparisonItem]]


class ModelEvaluation(BaseModel):
    model_name: str
    accuracy: float
    wrong_predictions: Dict[str, GroupedComparison]


class AdversarialExample(BaseModel):
    examples: List[str]
    predicted_label: str
    true_label: str
    task: str
