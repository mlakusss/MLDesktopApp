
import pytest

from services.prediction_service import PredictionService


def test_parse_valid_features():
    features = PredictionService.parse_features(
        "3.621, 8.666, -2.807, -0.446"
    )

    assert features == [3.621, 8.666, -2.807, -0.446]


@pytest.mark.parametrize(
    "raw_input",
    [
        "",
        "   ",
        "1, 2, 3",
        "1, 2, 3, 4, 5",
        "1, abc, 3, 4",
        "1, 2, NaN, 4",
        "1, 2, inf, 4",
        "1, 2, , 4",
    ]
)
def test_invalid_input(raw_input):
    with pytest.raises(ValueError):
        PredictionService.parse_features(raw_input)


def test_service_prediction():
    service = PredictionService("Logistic Regression")

    result = service.run(
        "3.621, 8.666, -2.807, -0.446"
    )

    assert result["prediction"] in [0, 1]
    assert result["class_name"] in [
        "Подлинная банкнота",
        "Поддельная банкнота",
    ]
    assert 0 <= result["confidence"] <= 1
    assert result["execution_time"] >= 0
    assert len(result["features"]) == 4


def test_model_switching():
    service = PredictionService("Logistic Regression")

    service.change_model("Random Forest")

    assert service.model_key == "Random Forest"

    result = service.run("3.621, 8.666, -2.807, -0.446")

    assert result["model_key"] == "Random Forest"


def test_unknown_model():
    service = PredictionService()

    with pytest.raises(ValueError):
        service.change_model("Unknown Model")
