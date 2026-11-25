
import pytest

from config import AVAILABLE_MODELS
from ml.predictor import Predictor


@pytest.mark.parametrize(
    "model_key",
    ["Logistic Regression", "Random Forest"]
)
def test_model_prediction(model_key):
    predictor = Predictor(
        AVAILABLE_MODELS[model_key]["path"]
    )

    result = predictor.predict(
        [3.621, 8.666, -2.807, -0.446]
    )

    assert result["prediction"] in [0, 1]
    assert 0 in result["probabilities"]
    assert 1 in result["probabilities"]

    total = sum(result["probabilities"].values())
    assert total == pytest.approx(1.0)


@pytest.mark.parametrize(
    "features",
    [
        [1, 2, 3],
        [1, 2, 3, 4, 5],
    ]
)
def test_wrong_number_of_features(features):
    predictor = Predictor(
        AVAILABLE_MODELS["Logistic Regression"]["path"]
    )

    with pytest.raises(ValueError):
        predictor.predict(features)


@pytest.mark.parametrize(
    "features",
    [
        [1, 2, float("nan"), 4],
        [1, 2, float("inf"), 4],
    ]
)
def test_invalid_numeric_values(features):
    predictor = Predictor(
        AVAILABLE_MODELS["Logistic Regression"]["path"]
    )

    with pytest.raises(ValueError):
        predictor.predict(features)
