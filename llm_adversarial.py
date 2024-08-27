from typing import Dict, List
from pydantic import BaseModel

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

GPT4_MODEL_NAME = "gpt-4o"
ANTHROPIC_MODEL_NAME = "claude-3-5-sonnet-20240620"
LLM_PROMPT = """
You are a helpful assistant that helps train machine learning models by identifying.
Your task is to explain the model incorrectly classified the data, and generate adversarial examples that would be misclassified by the model.
For the generated text data, provide the text data and the correct label, as well as the incorrect label that the model would predict.

Here is what you'll be provided as input:
- Task: A description of the task that the model should perform.
- Labels: A list of labels with their corresponding descriptions.
- Examples: A list of examples with the text data, the correct label, and the incorrect label.

Here is what you should provide as output (in JSON format):
- explanation: An explanation of why the model misclassified the examples and how the adversarial examples should be constructed.
- data: A list of adversarial examples with "text", "true_label", and "incorrect_label" fields.

NOTE:
- Length is extremely important to the model's classification, so ensure that the length of the text data is similar to the misclassified examples.
- Longer and more convoluted text data will help the model learn to classify more accurately, make the examples complex and difficult to classify.
- If there are grammatical errors in the example data, make sure you introduce similar errors in the adversarial examples, and exaggerate them to make the model misclassify the data.

For example,

{few_shot}

Task: {task}
Labels: {labels}
Examples: {examples}
Data:
"""

FEW_SHOT_EXAMPLES = """
Task: "Identify the language of the text data."
Labels:
{
    "ar": "Arabic",
    "bg": "Bulgarian",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sw": "Swahili",
    "th": "Thai",
    "tr": "Turkish",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese"
}
Examples:
{
    "data": [
        {
            text: "КАКВО ПРЕДСТАВЛЯВА СТАРШЕТО ПРЕДИЗВИКАТЕЛСТВО?",
            true_label: "bg",
            incorrect_label: "ru"
        },
        {
            text: "Не успях да намеря такова определение в речника.",
            true_label: "bg",
            incorrect_label: "ru"
        },
        {
            text: "Беше ми приятно да говорим.",
            true_label: "bg",
            incorrect_label: "ru"
        },
        {
            text: "Ще дойде ли Белият дом?",
            true_label: "bg",
            incorrect_label: "ru"
        },
        {
            text: "Необменни транзакции – печалби и загуби",
            true_label: "bg",
            incorrect_label: "ru"
        },
        {
            text: "Не можете да намерите по-евтин отговор.",
            true_label: "bg",
            incorrect_label: "ru"
        },
        {
            text: "Категорично. Негово благородие изчака момент за отговор.",
            true_label: "bg",
            incorrect_label: "ru"
        },
        {
            text: "Знаеш ли кой ще разбере?",
            true_label: "bg",
            incorrect_label: "ru"
        }
    ]
}
Data:
```json
{
    "explanation": "The text data is in Cyrillic script and uses many Russian words, which could confuse the model to classify it as Russian instead of Bulgarian. To help the model distinguish between the two similar Slavic languages, the adversarial examples should be mostly Russian, but with a few Bulgarian words or phrases to indicate the correct language.",
    "data": [
        {
            "text": "Момичето чете книга в парка.",
            "true_label": "bg",
            "incorrect_label": "ru"
        },
        {
            "text": "Кога ще пристигне влакът от София?",
            "true_label": "bg",
            "incorrect_label": "ru"
        },
        {
            "text": "Трябва да купя мляко и хляб от магазина.",
            "true_label": "bg",
            "incorrect_label": "ru"
        },
        {
            "text": "Той работи като програмист в голяма компания.",
            "true_label": "bg",
            "incorrect_label": "ru"
        },
        {
            "text": "Времето днес е слънчево и топло.",
            "true_label": "bg",
            "incorrect_label": "ru"
        },
        {
            "text": "Моля, затворете вратата, когато излизате.",
            "true_label": "bg",
            "incorrect_label": "ru"
        },
        {
            "text": "Децата играят футбол на игрището.",
            "true_label": "bg",
            "incorrect_label": "ru"
        },
        {
            "text": "Тя говори три езика свободно.",
            "true_label": "bg",
            "incorrect_label": "ru"
        },
        {
            "text": "Какво ще правиш през уикенда?",
            "true_label": "bg",
            "incorrect_label": "ru"
        },
        {
            "text": "Имам среща с лекаря в петък сутринта.",
            "true_label": "bg",
            "incorrect_label": "ru"
        }
    ]
}
```"""


# Types


class ClassificationItem(BaseModel):
    text: str
    true_label: str
    incorrect_label: str 

class AdversarialResponse(BaseModel):
    explanation: str
    data: List[ClassificationItem]
    


# Open AI

class AdversarialLlmOpenAI:
    def __init__(self):
        model = ChatOpenAI(
            **{
                "temperature": 0,
                "model_name": GPT4_MODEL_NAME,
            }
        ).bind(response_format={"type": "json_object"})
        parser = PydanticOutputParser(pydantic_object=AdversarialResponse)

        prompt = PromptTemplate(
            template=LLM_PROMPT,
            input_variables=["task", "examples"],
            partial_variables={"few_shot": FEW_SHOT_EXAMPLES},
        )

        self.generator = prompt | model | parser

    def generate(self, task: str, labels: Dict[str, str], examples: List[ClassificationItem]) -> AdversarialResponse:
        result: AdversarialResponse = self.generator.invoke(
            {
                "task": task,
                "labels": labels,
                "examples": examples
            }
        )
        return result
    
    async def agenerate(self, task: str, labels: Dict[str, str], examples: List[ClassificationItem]) -> AdversarialResponse:
        result: AdversarialResponse = await self.generator.ainvoke(
            {
                "task": task,
                "labels": labels,
                "examples": examples
            }
        )
        return result

# Anthropic

class AdversarialLlmAnthropic:
    def __init__(self):
        model = ChatAnthropic(
            **{
                "temperature": 0,
                "model_name": ANTHROPIC_MODEL_NAME,
            }
        )
        parser = PydanticOutputParser(pydantic_object=AdversarialResponse)

        prompt = PromptTemplate(
            template=LLM_PROMPT,
            input_variables=["task", "examples"],
            partial_variables={"few_shot": FEW_SHOT_EXAMPLES},
        )

        self.generator = prompt | model | parser

    def generate(self, task: str, labels: Dict[str, str], examples: List[ClassificationItem]) -> AdversarialResponse:
        result: AdversarialResponse = self.generator.invoke(
            {
                "task": task,
                "labels": labels,
                "examples": examples
            }
        )
        return result
    
    async def agenerate(self, task: str, labels: Dict[str, str], examples: List[ClassificationItem]) -> AdversarialResponse:
        result: AdversarialResponse = await self.generator.ainvoke(
            {
                "task": task,
                "labels": labels,
                "examples": examples
            }
        )
        return result