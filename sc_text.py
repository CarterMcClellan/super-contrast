from typing import List
from pydantic import BaseModel

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser

GPT4_MODEL_NAME = "gpt-4o"
TEXT_TREND_PROMPT = """You are a helpful assiant which generates adversarial tasks on a trend in the text data.

For example,

{few_shot}

Predicted Label: {pred_label} 
Correct Label: {true_label}

Data:
```
{examples}
```
Task:"""

TEXT_TREND_FS = """Predicted Label: positive_sentiment
Correct Label: negative_sentiment
Data:
```
[
    {
        "text": "This was not a good movie",
    },
    {
        "text": "Phil is not a good person",
    },
    {
        "text": "The food was not good",
    }
]
```
Task:
```json
{
    "task": "Generate text snippets that are misclassified as positive_sentiment but are actually negative_sentiment, because the model is not able to understand the negation in the text.
}
```"""


class TaskGeneratorResponse(BaseModel):
    task: str


class TaskGenerator:
    def __init__(self):
        llm = ChatOpenAI(
            **{
                "temperature": 0,
                "model_name": GPT4_MODEL_NAME,
            }
        ).bind(response_format={"type": "json_object"})
        parser = PydanticOutputParser(pydantic_object=TaskGeneratorResponse)

        prompt = PromptTemplate(
            template=TEXT_TREND_PROMPT,
            input_variables=["pred_label", "true_label", "examples"],
            partial_variables={"few_shot": TEXT_TREND_FS},
        )

        self.generator = prompt | llm | parser

    def generate(
        self, pred_label: str, true_label: str, examples: List[str]
    ) -> TaskGeneratorResponse:
        result: TaskGeneratorResponse = self.generator.invoke(
            {"pred_label": pred_label, "true_label": true_label, "examples": examples}
        )

        return result

ADVERSARIAL_PROMPT = """You are a helpful assistant which generates adversarial examples based on a trend in the text data.

For example

{few_shot}

Task: {task}
Example: {examples}
Data:
"""

ADVERSARIAL_EXAMPLES = """
Task: "Generate text snippets that are misclassified as Russian (ru) but are actually Bulgarian (bg), because the model is not able to distinguish between the two similar Slavic languages."
Examples: [
    "КАКВО ПРЕДСТАВЛЯВА СТАРШЕТО ПРЕДИЗВИКАТЕЛСТВО?",
    "Не успях да намеря такова определение в речника.",
    "Беше ми приятно да говорим.",
    "Ще дойде ли Белият дом?",
    "Необменни транзакции – печалби и загуби",
    "Не можете да намерите по-евтин отговор.",
    "Категорично. Негово благородие изчака момент за отговор.",
    "Знаеш ли кой ще разбере?"
]

Data:
```json
{
    "data": [
        {
            "text": "Момичето чете книга в парка.",
            "label": "bg"
        },
        {
            "text": "Кога ще пристигне влакът от София?",
            "label": "bg"
        },
        {
            "text": "Трябва да купя мляко и хляб от магазина.",
            "label": "bg"
        },
        {
            "text": "Той работи като програмист в голяма компания.",
            "label": "bg"
        },
        {
            "text": "Времето днес е слънчево и топло.",
            "label": "bg"
        },
        {
            "text": "Моля, затворете вратата, когато излизате.",
            "label": "bg"
        },
        {
            "text": "Децата играят футбол на игрището.",
            "label": "bg"
        },
        {
            "text": "Тя говори три езика свободно.",
            "label": "bg"
        },
        {
            "text": "Какво ще правиш през уикенда?",
            "label": "bg"
        },
        {
            "text": "Имам среща с лекаря в петък сутринта.",
            "label": "bg"
        }
    ]
}
```"""

class AdversarialItem(BaseModel):
    text: str
    label: str 

class AdversarialResponse(BaseModel):
    data: List[AdversarialItem]

class AdversarialGenerator:
    def __init__(self):
        llm = ChatOpenAI(
            **{
                "temperature": 0,
                "model_name": GPT4_MODEL_NAME,
            }
        ).bind(response_format={"type": "json_object"})
        parser = PydanticOutputParser(pydantic_object=AdversarialResponse)

        prompt = PromptTemplate(
            template=ADVERSARIAL_PROMPT,
            input_variables=["task", "examples"],
            partial_variables={"few_shot": ADVERSARIAL_EXAMPLES},
        )

        self.generator = prompt | llm | parser

    def generate(self, task: str, examples: List[AdversarialItem]) -> AdversarialResponse:
        result: AdversarialResponse = self.generator.invoke(
            {"task": task, "examples": examples}
        )

        return result