
import numpy as np
import pandas as pd

from ml.model_loader import ModelLoader


class Predictor:
    """Выполняет классификацию с помощью обученной модели."""

    def __init__(self, model_path):
        self.model = ModelLoader.load(model_path)

        if not hasattr(self.model, "predict"):
            raise TypeError(
                "Загруженный объект не поддерживает predict()."
            )

        if not hasattr(self.model, "predict_proba"):
            raise TypeError(
                "Модель не поддерживает predict_proba()."
            )

        self.expected_features = getattr(
            self.model,
            "n_features_in_",
            None
        )

        self.feature_names = getattr(
            self.model,
            "feature_names_in_",
            None
        )

    def predict(self, features):
        values = np.asarray(features, dtype=float)

        if values.ndim != 1:
            raise ValueError(
                "Ожидается одномерный список признаков."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Признаки должны быть конечными числами."
            )

        if (
            self.expected_features is not None
            and len(values) != self.expected_features
        ):
            raise ValueError(
                f"Ожидается {self.expected_features} признака, "
                f"получено {len(values)}."
            )

        if self.feature_names is not None:
            columns = list(self.feature_names)
        else:
            columns = [
                f"feature_{i}"
                for i in range(len(values))
            ]

        input_data = pd.DataFrame(
            [values],
            columns=columns
        )

        prediction = int(
            self.model.predict(input_data)[0]
        )

        probabilities = self.model.predict_proba(
            input_data
        )[0]

        return {
            "prediction": prediction,
            "probabilities": {
                int(class_id): float(probability)
                for class_id, probability
                in zip(self.model.classes_, probabilities)
            },
        }
