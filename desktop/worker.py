
from PySide6.QtCore import QThread, Signal

from services.prediction_service import PredictionService


class PredictWorker(QThread):
    """Фоновый поток для выполнения ML-предсказания."""

    result_ready = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, raw_input, model_key, parent=None):
        super().__init__(parent)

        self.raw_input = raw_input
        self.model_key = model_key

    def run(self):
        try:
            service = PredictionService(self.model_key)
            result = service.run(self.raw_input)
            self.result_ready.emit(result)
        except Exception as error:
            self.error_occurred.emit(str(error))
