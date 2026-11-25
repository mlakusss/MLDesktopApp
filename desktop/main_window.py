
from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFormLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QComboBox,
    QProgressBar,
    QGroupBox,
    QFrame,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
    QAbstractItemView,
)

from config import AVAILABLE_MODELS, FEATURE_NAMES
from desktop.worker import PredictWorker


class MainWindow(QMainWindow):
    """Главное окно приложения Banknote ML Analyzer."""

    def __init__(self):
        super().__init__()

        self.worker = None

        self.setWindowTitle("Banknote ML Analyzer")
        self.setMinimumSize(950, 720)
        self.resize(1100, 820)

        self._create_widgets()
        self._create_layout()
        self._create_menu()
        self._connect_signals()
        self._apply_styles()

    # -------------------------------------------------
    # СОЗДАНИЕ ВИДЖЕТОВ
    # -------------------------------------------------

    def _create_widgets(self):
        self.title_label = QLabel("Banknote ML Analyzer")
        self.title_label.setObjectName("titleLabel")

        self.subtitle_label = QLabel(
            "Интеллектуальная система определения подлинности банкнот"
        )
        self.subtitle_label.setObjectName("subtitleLabel")

        # Выбор модели
        self.model_combo = QComboBox()

        for model_key, model_info in AVAILABLE_MODELS.items():
            self.model_combo.addItem(
                model_info["name"],
                model_key
            )

        # Поля признаков
        self.feature_inputs = {}

        for feature in FEATURE_NAMES:
            field = QLineEdit()
            field.setPlaceholderText(
                f"Введите значение {feature}"
            )
            field.setClearButtonEnabled(True)
            self.feature_inputs[feature] = field

        # Кнопки
        self.predict_button = QPushButton("Предсказать")
        self.predict_button.setObjectName("primaryButton")

        self.clear_button = QPushButton("Очистить")
        self.clear_button.setObjectName("secondaryButton")

        # Результат
        self.result_label = QLabel(
            "Результат появится здесь"
        )
        self.result_label.setObjectName("resultLabel")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setWordWrap(True)

        self.confidence_label = QLabel("Уверенность: —")
        self.confidence_label.setObjectName("confidenceLabel")

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setFormat("%p%")

        # Вероятности классов
        self.class_0_label = QLabel("Подлинная банкнота: —")
        self.class_1_label = QLabel("Поддельная банкнота: —")

        self.class_0_bar = QProgressBar()
        self.class_1_bar = QProgressBar()

        for bar in (self.class_0_bar, self.class_1_bar):
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFormat("%p%")

        # Дополнительная информация
        self.execution_time_label = QLabel("Время выполнения: —")
        self.model_info_label = QLabel("Используемая модель: —")

        # Статус
        self.status_label = QLabel("Готово к работе")

        # История
        self.history_table = QTableWidget(0, 5)
        self.history_table.setHorizontalHeaderLabels([
            "Время",
            "Модель",
            "Результат",
            "Уверенность",
            "Время, мс",
        ])

        self.history_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.history_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.history_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.history_table.verticalHeader().setVisible(False)

        header = self.history_table.horizontalHeader()
        header.setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

    # -------------------------------------------------
    # РАЗМЕТКА ИНТЕРФЕЙСА
    # -------------------------------------------------

    def _create_layout(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(24, 20, 24, 16)
        main_layout.setSpacing(14)

        # Заголовок
        header_layout = QVBoxLayout()
        header_layout.addWidget(self.title_label)
        header_layout.addWidget(self.subtitle_label)

        main_layout.addLayout(header_layout)

        # Выбор модели
        model_group = QGroupBox("Модель машинного обучения")
        model_layout = QHBoxLayout(model_group)
        model_layout.addWidget(
            QLabel("Алгоритм классификации:")
        )
        model_layout.addWidget(self.model_combo, 1)

        main_layout.addWidget(model_group)

        # Ввод признаков
        input_group = QGroupBox("Входные данные банкноты")
        input_layout = QGridLayout(input_group)

        feature_descriptions = {
            "variance": "Дисперсия изображения",
            "skewness": "Асимметрия изображения",
            "curtosis": "Эксцесс изображения",
            "entropy": "Энтропия изображения",
        }

        for index, feature in enumerate(FEATURE_NAMES):
            row = index // 2
            column = (index % 2) * 2

            label = QLabel(
                feature_descriptions.get(feature, feature)
            )

            input_layout.addWidget(label, row, column)
            input_layout.addWidget(
                self.feature_inputs[feature],
                row,
                column + 1
            )

        main_layout.addWidget(input_group)

        # Кнопки
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.predict_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()

        main_layout.addLayout(button_layout)

        # Результат
        result_group = QGroupBox("Результат классификации")
        result_layout = QVBoxLayout(result_group)

        result_layout.addWidget(self.result_label)

        result_layout.addWidget(self.confidence_label)
        result_layout.addWidget(self.confidence_bar)

        result_layout.addSpacing(8)

        result_layout.addWidget(self.class_0_label)
        result_layout.addWidget(self.class_0_bar)

        result_layout.addWidget(self.class_1_label)
        result_layout.addWidget(self.class_1_bar)

        # Дополнительная информация
        info_layout = QHBoxLayout()
        info_layout.addWidget(self.execution_time_label)
        info_layout.addStretch()
        info_layout.addWidget(self.model_info_label)

        result_layout.addLayout(info_layout)

        main_layout.addWidget(result_group)

        # История
        history_group = QGroupBox("История предсказаний")
        history_layout = QVBoxLayout(history_group)
        history_layout.addWidget(self.history_table)

        main_layout.addWidget(history_group, 1)

        # Строка состояния
        self.statusBar().addWidget(self.status_label)

    # -------------------------------------------------
    # МЕНЮ ПРИЛОЖЕНИЯ
    # -------------------------------------------------

    def _create_menu(self):
        menu_bar = self.menuBar()

        # Файл
        file_menu = menu_bar.addMenu("Файл")

        clear_history_action = QAction(
            "Очистить историю",
            self
        )
        clear_history_action.triggered.connect(
            self._clear_history
        )

        exit_action = QAction("Выход", self)
        exit_action.triggered.connect(self.close)

        file_menu.addAction(clear_history_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        # Модель
        model_menu = menu_bar.addMenu("Модель")

        self.model_action_group = QActionGroup(self)
        self.model_action_group.setExclusive(True)

        self.model_actions = {}

        for index, (model_key, model_info) in enumerate(
            AVAILABLE_MODELS.items()
        ):
            action = QAction(
                model_info["name"],
                self
            )
            action.setCheckable(True)
            action.setData(model_key)

            if index == 0:
                action.setChecked(True)

            action.triggered.connect(
                lambda checked, key=model_key:
                self._select_model_from_menu(key)
            )

            self.model_action_group.addAction(action)
            model_menu.addAction(action)

            self.model_actions[model_key] = action

        # Справка
        help_menu = menu_bar.addMenu("Справка")

        about_action = QAction("О программе", self)
        about_action.triggered.connect(self._show_about)

        help_menu.addAction(about_action)

    # -------------------------------------------------
    # СОБЫТИЯ
    # -------------------------------------------------

    def _connect_signals(self):
        self.predict_button.clicked.connect(
            self._start_prediction
        )

        self.clear_button.clicked.connect(
            self._clear_inputs
        )

        self.model_combo.currentIndexChanged.connect(
            self._on_model_changed
        )

    def _select_model_from_menu(self, model_key):
        index = self.model_combo.findData(model_key)

        if index >= 0:
            self.model_combo.setCurrentIndex(index)

    def _on_model_changed(self, index):
        model_key = self.model_combo.itemData(index)

        if model_key in self.model_actions:
            self.model_actions[model_key].setChecked(True)

        self.status_label.setText(
            f"Выбрана модель: "
            f"{AVAILABLE_MODELS[model_key]['name']}"
        )

    # -------------------------------------------------
    # ПРЕДСКАЗАНИЕ В ОТДЕЛЬНОМ ПОТОКЕ
    # -------------------------------------------------

    def _start_prediction(self):
        if self.worker is not None:
            return

        values = [
            self.feature_inputs[feature].text().strip()
            for feature in FEATURE_NAMES
        ]

        raw_input = ", ".join(values)

        model_key = self.model_combo.currentData()

        self.predict_button.setEnabled(False)
        self.model_combo.setEnabled(False)

        self.result_label.setText(
            "Выполняется классификация..."
        )
        self.result_label.setStyleSheet(
            "color: #64748b; font-weight: bold;"
        )

        self.status_label.setText(
            "Вычисление выполняется в отдельном потоке..."
        )

        self.worker = PredictWorker(
            raw_input,
            model_key,
            self
        )

        self.worker.result_ready.connect(
            self._on_prediction_success
        )

        self.worker.error_occurred.connect(
            self._on_prediction_error
        )

        self.worker.finished.connect(
            self._on_worker_finished
        )

        self.worker.finished.connect(
            self.worker.deleteLater
        )

        self.worker.start()


    def _on_prediction_success(self, result):
        prediction = result["prediction"]
        confidence = result["confidence"]

        # Основной результат
        self.result_label.setText(
            result["class_name"]
        )

        if prediction == 0:
            self.result_label.setStyleSheet(
                "color: #15803d; font-weight: bold;"
            )
        else:
            self.result_label.setStyleSheet(
                "color: #dc2626; font-weight: bold;"
            )

        # Уверенность
        self.confidence_label.setText(
            f"Уверенность модели: {confidence * 100:.4f}%"
        )

        self.confidence_bar.setValue(
            round(confidence * 100)
        )

        # Вероятности классов
        probability_0 = result["probabilities"][0]
        probability_1 = result["probabilities"][1]

        self.class_0_label.setText(
            f"Подлинная банкнота: {probability_0 * 100:.4f}%"
        )

        self.class_1_label.setText(
            f"Поддельная банкнота: {probability_1 * 100:.4f}%"
        )

        self.class_0_bar.setValue(
            round(probability_0 * 100)
        )

        self.class_1_bar.setValue(
            round(probability_1 * 100)
        )

        # Информация о модели
        self.execution_time_label.setText(
            f"Время выполнения: "
            f"{result['execution_time']:.3f} мс"
        )

        self.model_info_label.setText(
            f"Модель: {result['model_name']}"
        )

        self.status_label.setText(
            "Классификация успешно завершена"
        )

        self._add_history_record(result)

    def _on_prediction_error(self, message):
        self.result_label.setText(
            "Не удалось выполнить предсказание"
        )

        self.result_label.setStyleSheet(
            "color: #dc2626; font-weight: bold;"
        )

        self.status_label.setText(
            "Ошибка классификации"
        )

        QMessageBox.warning(
            self,
            "Ошибка ввода или вычисления",
            message
        )

    def _on_worker_finished(self):
        self.predict_button.setEnabled(True)
        self.model_combo.setEnabled(True)

        self.worker = None

    # -------------------------------------------------
    # ИСТОРИЯ
    # -------------------------------------------------

    def _add_history_record(self, result):
        row = self.history_table.rowCount()

        self.history_table.insertRow(row)

        timestamp = datetime.now().strftime("%H:%M:%S")

        values = [
            timestamp,
            result["model_name"],
            result["class_name"],
            f"{result['confidence'] * 100:.2f}%",
            f"{result['execution_time']:.3f}",
        ]

        for column, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter
            )
            self.history_table.setItem(
                row,
                column,
                item
            )

        self.history_table.scrollToBottom()

    def _clear_history(self):
        self.history_table.setRowCount(0)

        self.status_label.setText(
            "История предсказаний очищена"
        )

    # -------------------------------------------------
    # ОЧИСТКА ИНТЕРФЕЙСА
    # -------------------------------------------------

    def _clear_inputs(self):
        if self.worker is not None:
            return

        for field in self.feature_inputs.values():
            field.clear()

        self.result_label.setText(
            "Результат появится здесь"
        )

        self.result_label.setStyleSheet("")

        self.confidence_label.setText(
            "Уверенность: —"
        )

        self.confidence_bar.setValue(0)

        self.class_0_label.setText(
            "Подлинная банкнота: —"
        )

        self.class_1_label.setText(
            "Поддельная банкнота: —"
        )

        self.class_0_bar.setValue(0)
        self.class_1_bar.setValue(0)

        self.execution_time_label.setText(
            "Время выполнения: —"
        )

        self.model_info_label.setText(
            "Используемая модель: —"
        )

        self.status_label.setText(
            "Поля ввода очищены"
        )

    # -------------------------------------------------
    # СПРАВКА
    # -------------------------------------------------

    def _show_about(self):
        QMessageBox.about(
            self,
            "О программе",
            """
            <h2>Banknote ML Analyzer</h2>

            <p>Десктопное приложение для определения
            подлинности банкнот с использованием
            методов машинного обучения.</p>

            <p><b>Используемые алгоритмы:</b></p>
            <ul>
                <li>Logistic Regression</li>
                <li>Random Forest</li>
            </ul>

            <p><b>Технологии:</b></p>
            <ul>
                <li>Python 3.12</li>
                <li>PySide6</li>
                <li>scikit-learn</li>
                <li>NumPy и Pandas</li>
            </ul>

            <p>Практическая работа №3</p>
            """
        )

    # -------------------------------------------------
    # ОФОРМЛЕНИЕ
    # -------------------------------------------------

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f1f5f9;
            }

            QWidget {
                color: #1e293b;
                font-family: "Segoe UI";
                font-size: 10pt;
            }

            QLabel#titleLabel {
                color: #0f172a;
                font-size: 23pt;
                font-weight: bold;
            }

            QLabel#subtitleLabel {
                color: #64748b;
                font-size: 10pt;
            }

            QGroupBox {
                background-color: white;
                border: 1px solid #dbe3ee;
                border-radius: 10px;
                margin-top: 12px;
                padding: 14px;
                font-weight: bold;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 5px;
                color: #334155;
            }

            QLineEdit, QComboBox {
                background-color: #f8fafc;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 9px;
                min-height: 22px;
            }

            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #2563eb;
            }

            QPushButton {
                border: none;
                border-radius: 7px;
                padding: 10px 20px;
                font-weight: bold;
            }

            QPushButton#primaryButton {
                background-color: #2563eb;
                color: white;
            }

            QPushButton#primaryButton:hover {
                background-color: #1d4ed8;
            }

            QPushButton#primaryButton:disabled {
                background-color: #94a3b8;
            }

            QPushButton#secondaryButton {
                background-color: #e2e8f0;
                color: #334155;
            }

            QPushButton#secondaryButton:hover {
                background-color: #cbd5e1;
            }

            QLabel#resultLabel {
                font-size: 19pt;
                font-weight: bold;
                padding: 12px;
            }

            QLabel#confidenceLabel {
                font-weight: bold;
            }

            QProgressBar {
                border: none;
                border-radius: 5px;
                background-color: #e2e8f0;
                height: 15px;
                text-align: center;
                font-size: 8pt;
            }

            QProgressBar::chunk {
                background-color: #2563eb;
                border-radius: 5px;
            }

            QTableWidget {
                background-color: white;
                alternate-background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                gridline-color: #e2e8f0;
            }

            QHeaderView::section {
                background-color: #e2e8f0;
                color: #334155;
                padding: 8px;
                border: none;
                font-weight: bold;
            }

            QStatusBar {
                background-color: #e2e8f0;
                color: #475569;
            }

            QMenuBar {
                background-color: white;
            }

            QMenuBar::item {
                padding: 7px 12px;
            }

            QMenuBar::item:selected {
                background-color: #dbeafe;
            }

            QMenu {
                background-color: white;
                border: 1px solid #cbd5e1;
            }

            QMenu::item {
                padding: 7px 25px;
            }

            QMenu::item:selected {
                background-color: #dbeafe;
            }
        """)

    # -------------------------------------------------
    # ЗАКРЫТИЕ ПРИЛОЖЕНИЯ
    # -------------------------------------------------

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(
                self,
                "Выполняется операция",
                "Дождитесь завершения классификации перед закрытием приложения."
            )
            event.ignore()
            return

        event.accept()
