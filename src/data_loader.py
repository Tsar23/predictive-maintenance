"""Завантаження набору даних AI4I 2020 Predictive Maintenance."""

import io
import zipfile

import pandas as pd

from src import config


def download_dataset():
    """Завантажує набір даних з UCI та зберігає його у data/ai4i2020.csv.

    Повертає кортеж (успіх, повідомлення).
    """
    import requests  # встановлюється разом зі streamlit

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in config.DATA_URLS:
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            content = response.content
            if url.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(content)) as archive:
                    csv_names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
                    if not csv_names:
                        raise ValueError("в архіві немає CSV-файлу")
                    content = archive.read(csv_names[0])
            config.DATA_FILE.write_bytes(content)
            return True, "Набір даних успішно завантажено з UCI Machine Learning Repository."
        except Exception as exc:  # пробуємо наступне джерело
            errors.append(f"{url}: {exc}")
    return False, "Не вдалося завантажити дані:\n" + "\n".join(errors)


def read_raw(source=None):
    """Зчитує CSV-файл (шлях або завантажений файл) і перейменовує колонки."""
    source = source if source is not None else config.DATA_FILE
    df = pd.read_csv(source)
    df = df.rename(columns=config.COLUMN_MAP)
    missing = [c for c in config.COLUMN_MAP.values() if c not in df.columns]
    if missing:
        raise ValueError("У файлі відсутні обов'язкові колонки: " + ", ".join(missing))
    return df


def dataset_exists():
    return config.DATA_FILE.exists()
