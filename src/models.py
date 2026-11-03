"""Побудова, навчання, оцінювання та збереження моделей класифікації."""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.preprocessing import failure_type_label

RS = config.RANDOM_STATE

ALGORITHMS = {
    "Випадковий ліс (Random Forest)": lambda: RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample",
        random_state=RS, n_jobs=-1),
    "Дерево рішень (Decision Tree)": lambda: DecisionTreeClassifier(
        max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=RS),
    "Градієнтний бустинг (Gradient Boosting)": lambda: GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1, random_state=RS),
    "Логістична регресія (Logistic Regression)": lambda: LogisticRegression(
        max_iter=2000, class_weight="balanced"),
    "k-найближчих сусідів (k-NN)": lambda: KNeighborsClassifier(n_neighbors=7, weights="distance"),
}
DEFAULT_ALGORITHM = "Випадковий ліс (Random Forest)"


def build_pipeline(algorithm, scale=True):
    """Конвеєр: стандартизація ознак (z-оцінка) + класифікатор."""
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", ALGORITHMS[algorithm]()))
    return Pipeline(steps)


def split_data(df, features, test_size=0.25):
    X = df[features]
    y = df[config.TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RS)


def compute_metrics(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, proba),
    }


def train_model(df, features, algorithm=DEFAULT_ALGORITHM, test_size=0.25,
                scale=True, threshold=0.5):
    """Навчає модель і повертає словник з усіма результатами оцінювання."""
    X_train, X_test, y_train, y_test = split_data(df, features, test_size)
    pipe = build_pipeline(algorithm, scale)
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_test, proba)

    # 5-кратна стратифікована крос-валідація на всій вибірці
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    cv_res = cross_validate(build_pipeline(algorithm, scale), df[features], df[config.TARGET],
                            cv=cv, scoring=["f1", "recall", "precision", "roc_auc"])
    cv_summary = {k.replace("test_", ""): (float(np.mean(v)), float(np.std(v)))
                  for k, v in cv_res.items() if k.startswith("test_")}

    # Важливість ознак методом перестановок (не залежить від типу моделі)
    perm = permutation_importance(pipe, X_test, y_test, scoring="roc_auc",
                                  n_repeats=5, random_state=RS, n_jobs=-1)
    importance = pd.Series(perm.importances_mean, index=features).sort_values(ascending=False)

    type_model = train_type_model(df, features)

    return {
        "pipeline": pipe,
        "type_model": type_model,
        "algorithm": algorithm,
        "features": features,
        "scale": scale,
        "threshold": threshold,
        "test_size": test_size,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "metrics": compute_metrics(y_test, proba, threshold),
        "confusion": confusion_matrix(y_test, pred),
        "roc": (fpr, tpr),
        "cv": cv_summary,
        "importance": importance,
        "y_test": y_test.to_numpy(),
        "proba": proba,
    }


def compare_algorithms(df, features, test_size=0.25, scale=True, threshold=0.5):
    """Порівнює всі алгоритми на однаковому розбитті даних."""
    X_train, X_test, y_train, y_test = split_data(df, features, test_size)
    rows = []
    for name in ALGORITHMS:
        pipe = build_pipeline(name, scale)
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        row = {"Алгоритм": name}
        row.update(compute_metrics(y_test, proba, threshold))
        rows.append(row)
    table = pd.DataFrame(rows).set_index("Алгоритм")
    return table.sort_values("F1", ascending=False)


def train_type_model(df, features):
    """Додаткова модель: визначає найімовірніший тип несправності."""
    failed = df[df[config.TARGET] == 1]
    if failed.empty:
        return None
    labels = failed.apply(failure_type_label, axis=1)
    model = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                   random_state=RS, n_jobs=-1)
    model.fit(failed[features], labels)
    return model


def extract_rules(df, features, depth=3):
    """Будує неглибоке дерево рішень і повертає зрозумілі людині правила."""
    from sklearn.tree import export_text
    tree = DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=RS)
    tree.fit(df[features], df[config.TARGET])
    names = [config.LABELS.get(f, f) for f in features]
    text = export_text(tree, feature_names=names, decimals=1, show_weights=False)
    return text.replace("class: 1", "НЕСПРАВНІСТЬ").replace("class: 0", "норма")


def failure_rate_by_bins(df, column, bins=5):
    """Частота несправностей у межах інтервалів значень ознаки."""
    groups = pd.qcut(df[column], q=bins, duplicates="drop")
    table = df.groupby(groups, observed=True)[config.TARGET].agg(["count", "mean"])
    low = df[column].min()
    table.index = [f"{max(iv.left, low):.1f} – {iv.right:.1f}" for iv in table.index]
    table.columns = ["Кількість записів", "Частка несправностей, %"]
    table["Частка несправностей, %"] = (table["Частка несправностей, %"] * 100).round(2)
    return table


def risk_level(probability):
    if probability < 0.2:
        return "Низький"
    if probability < 0.5:
        return "Середній"
    return "Високий"


def predict(result, new_df):
    """Класифікує нові об'єкти (одне або кілька одиниць обладнання)."""
    X = new_df[result["features"]]
    proba = result["pipeline"].predict_proba(X)[:, 1]
    out = pd.DataFrame({
        "Ймовірність несправності": np.round(proba, 4),
        "Прогноз": np.where(proba >= result["threshold"], "Несправність", "Норма"),
        "Рівень ризику": [risk_level(p) for p in proba],
    }, index=new_df.index)
    if result["type_model"] is not None:
        types = result["type_model"].predict(X)
        out["Ймовірний тип несправності"] = [
            config.FAILURE_TYPES[t] if p >= result["threshold"] else "—"
            for t, p in zip(types, proba)]
    return out


def type_probabilities(result, row_df):
    model = result["type_model"]
    if model is None:
        return pd.Series(dtype=float)
    proba = model.predict_proba(row_df[result["features"]])[0]
    return pd.Series(proba, index=model.classes_).sort_values(ascending=False)


def save_model(result, path=config.MODEL_FILE):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, path)
    return path


def load_model(path=config.MODEL_FILE):
    if not path.exists():
        return None
    return joblib.load(path)
