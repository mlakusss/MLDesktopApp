
import time
import math

from config import AVAILABLE_MODELS, CLASS_NAMES, FEATURE_NAMES
from ml.predictor import Predictor


class PredictionService:
    """Сервис для обработки запросов на классификацию банкнот."""

    def __init__(self, model_key="Logistic Regression"):
        self.model_key = model_key
        self.predictor = None
        self._load_model(model_key)

    def _load_model(self, model_key):
        """Загружает выбранную модель."""

        if model_key not in AVAILABLE_MODELS:
            raise ValueError(
                f"Неизвестная модель: {model_key}"
            )

        model_path = AVAILABLE_MODELS[model_key]["path"]
        self.predictor = Predictor(model_path)
        self.model_key = model_key

    def change_model(self, model_key):
        """Переключает используемую модель."""

        if model_key == self.model_key:
            return

        self._load_model(model_key)

    @staticmethod
    def parse_features(raw_input):
        """Преобразует строку признаков в список чисел."""

        if not raw_input or not raw_input.strip():
            raise ValueError(
                "Введите значения признаков."
            )

        parts = raw_input.split(",")

        if len(parts) != len(FEATURE_NAMES):
            raise ValueError(
                f"Необходимо ввести ровно "
                f"{len(FEATURE_NAMES)} признака через запятую."
            )

        features = []

        for index, part in enumerate(parts):
            value = part.strip()

            if not value:
                raise ValueError(
                    f"Не указано значение признака "
                    f"'{FEATURE_NAMES[index]}'."
                )

            try:
                number = float(value)
            except ValueError:
                raise ValueError(
                    f"Значение признака "
                    f"'{FEATURE_NAMES[index]}' должно быть числом."
                )

            if not math.isfinite(number):
                raise ValueError(
                    f"Значение признака "
                    f"'{FEATURE_NAMES[index]}' должно быть конечным числом."
                )

            features.append(number)

        return features

    def run(self, raw_input):
        """Выполняет полный цикл классификации."""

        features = self.parse_features(raw_input)

        start_time = time.perf_counter()

        prediction_result = self.predictor.predict(features)

        execution_time = (
            time.perf_counter() - start_time
        ) * 1000

        prediction = prediction_result["prediction"]
        probabilities = prediction_result["probabilities"]

        confidence = max(probabilities.values())

        return {
            "prediction": prediction,
            "class_name": CLASS_NAMES[prediction],
            "probabilities": probabilities,
            "confidence": confidence,
            "execution_time": execution_time,
            "model_key": self.model_key,
            "model_name": AVAILABLE_MODELS[self.model_key]["name"],
            "features": features,
        }
