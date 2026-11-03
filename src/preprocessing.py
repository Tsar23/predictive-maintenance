"""Оцінювання якості даних, очищення, формування ознак та їх відбір за кореляцією."""

import numpy as np
import pandas as pd

from src import config

FAILURE_FLAGS = list(config.FAILURE_TYPES.keys())

# Фізично допустимі межі значень (для виявлення помилкових записів)
VALID_RANGES = {
    "air_temp": (250, 350),
    "process_temp": (250, 400),
    "rot_speed": (0, 5000),
    "torque": (0, 150),
    "tool_wear": (0, 400),
}


def quality_report(df):
    """Формує звіт про якість вхідних даних."""
    report = {
        "rows": len(df),
        "columns": df.shape[1],
        "missing": df.isna().sum(),
        "duplicates": int(df.duplicated().sum()),
        "duplicate_ids": int(df["udi"].duplicated().sum()),
    }

    invalid = {}
    for col, (low, high) in VALID_RANGES.items():
        invalid[col] = int(((df[col] < low) | (df[col] > high)).sum())
    report["invalid"] = pd.Series(invalid)

    outliers = {}
    for col in config.NUMERIC_RAW:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
        outliers[col] = int(mask.sum())
    report["outliers"] = pd.Series(outliers)

    any_flag = df[FAILURE_FLAGS].sum(axis=1) > 0
    report["failure_without_type"] = int(((df["failure"] == 1) & ~any_flag).sum())
    report["type_without_failure"] = int(((df["failure"] == 0) & any_flag).sum())
    report["unknown_type"] = int((~df["type"].isin(config.TYPE_CODES.keys())).sum())
    return report


def clean_data(df):
    """Очищує дані. Повертає очищений DataFrame та журнал виконаних дій."""
    log = []
    data = df.copy()
    start = len(data)

    before = len(data)
    data = data.dropna()
    log.append(("Видалено записи з пропущеними значеннями", before - len(data)))

    before = len(data)
    data = data.drop_duplicates()
    data = data.drop_duplicates(subset="udi")
    log.append(("Видалено дублікати", before - len(data)))

    before = len(data)
    for col, (low, high) in VALID_RANGES.items():
        data = data[(data[col] >= low) & (data[col] <= high)]
    data = data[data["type"].isin(config.TYPE_CODES.keys())]
    log.append(("Видалено записи з фізично неможливими значеннями", before - len(data)))

    # Суперечливі записи: несправність без типу або тип без несправності
    before = len(data)
    any_flag = data[FAILURE_FLAGS].sum(axis=1) > 0
    data = data[~(((data["failure"] == 1) & ~any_flag) | ((data["failure"] == 0) & any_flag))]
    log.append(("Видалено суперечливі записи (мітка несправності не узгоджена з її типом)",
                before - len(data)))

    # Ідентифікатори не несуть інформації для моделі
    data = data.drop(columns=["udi", "product_id"])
    log.append(("Вилучено колонки-ідентифікатори (UDI, Product ID)", 0))
    log.append(("Разом залишилось записів", len(data)))
    log.append(("Разом видалено записів", start - len(data)))
    return data.reset_index(drop=True), log


def add_features(df):
    """Додає похідні ознаки, що мають фізичний зміст."""
    data = df.copy()
    data["type_code"] = data["type"].map(config.TYPE_CODES)
    # Різниця температур процесу та повітря відповідає за тепловідведення
    data["temp_diff"] = data["process_temp"] - data["air_temp"]
    # Механічна потужність P = M * w, де w = 2*pi*n/60
    data["power"] = data["torque"] * data["rot_speed"] * 2 * np.pi / 60
    return data


CANDIDATE_FEATURES = ["type_code", "air_temp", "process_temp", "rot_speed",
                      "torque", "tool_wear", "temp_diff", "power"]


def correlation_matrix(df, columns=None):
    columns = columns or CANDIDATE_FEATURES + [config.TARGET]
    return df[columns].corr(method="pearson")


def strong_pairs(corr, threshold=config.CORR_THRESHOLD):
    """Пари ознак з модулем коефіцієнта кореляції не менше порогу."""
    features = [c for c in corr.columns if c != config.TARGET]
    pairs = []
    for i, a in enumerate(features):
        for b in features[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) >= threshold:
                pairs.append((a, b, round(float(r), 3)))
    return pairs


def vif(df, features):
    """Коефіцієнти інфляції дисперсії VIF = 1 / (1 - R²) для кожної ознаки."""
    X = df[features].to_numpy(dtype=float)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    result = {}
    for j, feat in enumerate(features):
        y = X[:, j]
        others = np.delete(X, j, axis=1)
        A = np.column_stack([np.ones(len(y)), others])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        r2 = 1 - resid.var() / y.var()
        result[feat] = np.inf if r2 >= 1 - 1e-9 else 1 / (1 - r2)
    return pd.Series(result)


def select_features(corr, threshold=config.CORR_THRESHOLD, df=None,
                    vif_threshold=config.VIF_THRESHOLD):
    """Відбір ознак у два етапи.

    1) Ознаки впорядковуються за силою зв'язку з цільовою змінною; ознака
       залишається, якщо вона не має сильної кореляції з жодною з уже відібраних.
    2) Якщо передано дані (df), перевіряється мультиколінеарність: поки
       найбільший VIF перевищує поріг, вилучається ознака з найбільшим VIF
       (серед рівних — слабше пов'язана з цільовою змінною).
    Повертає (відібрані ознаки, словник {вилучена ознака: пояснення}).
    """
    features = [c for c in corr.columns if c != config.TARGET]
    target_corr = corr[config.TARGET].drop(config.TARGET).abs()
    order = target_corr.sort_values(ascending=False).index
    selected, dropped = [], {}
    for feat in order:
        conflict = [s for s in selected if abs(corr.loc[feat, s]) >= threshold]
        if conflict:
            dropped[feat] = (f"сильна кореляція з ознакою «{config.LABELS.get(conflict[0])}» "
                             f"(r = {corr.loc[feat, conflict[0]]:.3f})")
        else:
            selected.append(feat)

    if df is not None:
        while len(selected) > 2:
            v = vif(df, selected)
            if v.max() <= vif_threshold:
                break
            worst = v[v >= v.max() - 1e-6].index
            feat = min(worst, key=lambda f: target_corr[f])
            value = "∞" if np.isinf(v[feat]) else f"{v[feat]:.1f}"
            dropped[feat] = (f"мультиколінеарність (VIF = {value}): ознака лінійно "
                             "виражається через інші відібрані ознаки")
            selected.remove(feat)

    selected = [f for f in features if f in selected]  # зберігаємо початковий порядок
    return selected, dropped


def failure_type_label(row):
    """Основний тип несправності для запису з несправністю."""
    for flag in FAILURE_FLAGS:
        if row[flag] == 1:
            return flag
    return "RNF"
