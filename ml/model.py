from dataclasses import dataclass
from pathlib import Path

import yaml
from transformers import pipeline

# load config file
# = Path(__file__).parent / "config.yaml"
#with open(config_path, "r") as file:
    #config = yaml.load(file, Loader=yaml.FullLoader)


@dataclass
class SentimentPrediction:
    """Class representing a sentiment prediction result."""

    label: str
    score: float


def load_model():
    """Load a pre-trained spam classification model.
    """

    model_hf = pipeline('text-classification', model='skandavivek2/spam-classifier', device=-1)

    def model(text: str) -> SentimentPrediction:
        pred = model_hf(text)
        pred_best_class = pred[0]
        return SentimentPrediction(
            label=pred_best_class["label"],
            score=pred_best_class["score"],
        )

    return model
