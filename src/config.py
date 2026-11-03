"""Загальні налаштування застосунку: шляхи, назви колонок, параметри."""

from pathlib import Path

# Кореневі каталоги проєкту
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DATA_FILE = DATA_DIR / "ai4i2020.csv"
MODEL_FILE = MODELS_DIR / "failure_model.joblib"

# Джерела набору даних AI4I 2020 Predictive Maintenance (UCI Machine Learning Repository)
DATA_URLS = [
    "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv",
]
DATA_PAGE = "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset"

# Перейменування колонок у короткі технічні назви
COLUMN_MAP = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "process_temp",
    "Rotational speed [rpm]": "rot_speed",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": "failure",
    "TWF": "TWF",
    "HDF": "HDF",
    "PWF": "PWF",
    "OSF": "OSF",
    "RNF": "RNF",
}

# Українські підписи ознак для інтерфейсу та графіків
LABELS = {
    "type": "Клас якості продукції",
    "type_code": "Клас якості продукції",
    "air_temp": "Температура повітря, K",
    "process_temp": "Температура процесу, K",
    "rot_speed": "Швидкість обертання, об/хв",
    "torque": "Крутний момент, Н·м",
    "tool_wear": "Знос інструмента, хв",
    "temp_diff": "Різниця температур, K",
    "power": "Механічна потужність, Вт",
    "failure": "Несправність",
}

# Типи несправностей
FAILURE_TYPES = {
    "TWF": "Знос інструмента (Tool Wear Failure)",
    "HDF": "Недостатнє тепловідведення (Heat Dissipation Failure)",
    "PWF": "Збій потужності (Power Failure)",
    "OSF": "Механічне перевантаження (Overstrain Failure)",
    "RNF": "Випадкова несправність (Random Failure)",
}

RECOMMENDATIONS = {
    "TWF": "Замінити або переточити ріжучий інструмент, скоротити інтервал планової заміни.",
    "HDF": "Перевірити систему охолодження та вентиляцію, знизити температуру процесу.",
    "PWF": "Перевірити привід і режим роботи: потужність виходить за допустимі межі.",
    "OSF": "Зменшити навантаження (крутний момент) або замінити зношений інструмент.",
    "RNF": "Провести загальний технічний огляд обладнання.",
}

TYPE_CODES = {"L": 0, "M": 1, "H": 2}
TYPE_NAMES = {"L": "L — низька", "M": "M — середня", "H": "H — висока"}

NUMERIC_RAW = ["air_temp", "process_temp", "rot_speed", "torque", "tool_wear"]
TARGET = "failure"

# Поріг сильної кореляції (за модулем)
CORR_THRESHOLD = 0.8
RANDOM_STATE = 42
