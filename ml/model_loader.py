
import joblib
from pathlib import Path


class ModelLoader:
    """Отвечает за загрузку обученных ML-моделей."""

    @staticmethod
    def load(model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Файл модели не найден: {model_path}"
            )

        try:
            return joblib.load(model_path)
        except Exception as error:
            raise RuntimeError(
                f"Не удалось загрузить модель: {error}"
            ) from error
