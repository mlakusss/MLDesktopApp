
from pathlib import Path

# Корневая директория проекта
BASE_DIR = Path(__file__).resolve().parent

# Директория с обученными моделями
MODELS_DIR = BASE_DIR / "models"

# Доступные модели
AVAILABLE_MODELS = {
    "Logistic Regression": {
        "name": "Логистическая регрессия",
        "path": MODELS_DIR / "logistic_regression.joblib",
        "description": "Линейная модель классификации",
    },
    "Random Forest": {
        "name": "Случайный лес",
        "path": MODELS_DIR / "random_forest.joblib",
        "description": "Ансамбль деревьев решений",
    },
}

# Названия признаков датасета
FEATURE_NAMES = [
    "variance",
    "skewness",
    "curtosis",
    "entropy",
]

# Количество признаков
EXPECTED_FEATURES = len(FEATURE_NAMES)

# Названия классов
CLASS_NAMES = {
    0: "Подлинная банкнота",
    1: "Поддельная банкнота",
}
