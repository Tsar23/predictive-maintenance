"""Побудова графіків (matplotlib) для інтерфейсу застосунку."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import config  # noqa: E402

BLUE = "#2563eb"
RED = "#dc2626"
GREY = "#94a3b8"
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
})


def label(col):
    return config.LABELS.get(col, col)


def class_balance(df):
    counts = df[config.TARGET].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    bars = ax.bar(["Норма (0)", "Несправність (1)"], counts.values, color=[BLUE, RED])
    for bar, value in zip(bars, counts.values):
        share = value / counts.sum() * 100
        ax.annotate(f"{value} ({share:.1f}%)", (bar.get_x() + bar.get_width() / 2, value),
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Кількість записів")
    ax.set_title("Розподіл цільової змінної")
    fig.tight_layout()
    return fig


def failure_types(df):
    counts = df[list(config.FAILURE_TYPES)].sum().sort_values()
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.barh(counts.index, counts.values, color=RED)
    for i, value in enumerate(counts.values):
        ax.annotate(str(int(value)), (value, i), va="center", xytext=(3, 0),
                    textcoords="offset points", fontsize=9)
    ax.set_xlabel("Кількість випадків")
    ax.set_title("Кількість несправностей за типами")
    fig.tight_layout()
    return fig


def distributions(df, columns):
    n = len(columns)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, columns):
        ok = df[df[config.TARGET] == 0][col]
        bad = df[df[config.TARGET] == 1][col]
        ax.hist(ok, bins=40, color=BLUE, alpha=0.6, density=True, label="Норма")
        ax.hist(bad, bins=40, color=RED, alpha=0.6, density=True, label="Несправність")
        ax.set_title(label(col), fontsize=10)
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].legend(fontsize=8)
    fig.suptitle("Розподіл ознак для справного та несправного обладнання")
    fig.tight_layout()
    return fig


def boxplots(df, columns):
    fig, axes = plt.subplots(1, len(columns), figsize=(2.3 * len(columns), 3.4))
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, columns):
        ax.boxplot(df[col].dropna(), patch_artist=True,
                   boxprops={"facecolor": "#dbeafe", "color": BLUE},
                   medianprops={"color": RED}, flierprops={"markersize": 2})
        ax.set_title(label(col), fontsize=8)
        ax.set_xticks([])
    fig.suptitle("Діаграми розмаху (виявлення викидів)")
    fig.tight_layout()
    return fig


def correlation_heatmap(corr):
    names = [label(c) for c in corr.columns]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Матриця кореляції Пірсона")
    fig.tight_layout()
    return fig


def scatter_pair(df, a, b):
    fig, ax = plt.subplots(figsize=(5, 3.8))
    sample = df.sample(min(len(df), 3000), random_state=config.RANDOM_STATE)
    ok = sample[sample[config.TARGET] == 0]
    bad = df[df[config.TARGET] == 1]
    ax.scatter(ok[a], ok[b], s=5, alpha=0.4, color=BLUE, label="Норма")
    ax.scatter(bad[a], bad[b], s=10, alpha=0.8, color=RED, label="Несправність")
    ax.set_xlabel(label(a))
    ax.set_ylabel(label(b))
    ax.legend(fontsize=8)
    ax.set_title("Діаграма розсіювання")
    fig.tight_layout()
    return fig


def confusion(cm):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    ticks = ["Норма", "Несправність"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)
    ax.set_xlabel("Прогноз моделі")
    ax.set_ylabel("Фактичний стан")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Матриця помилок")
    fig.tight_layout()
    return fig


def roc(fpr, tpr, auc):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"ROC-крива (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="Випадкове вгадування")
    ax.set_xlabel("Частка хибнопозитивних (FPR)")
    ax.set_ylabel("Частка істиннопозитивних (TPR)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("ROC-крива")
    fig.tight_layout()
    return fig


def importance(series):
    s = series.sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    colors = [RED if v == s.max() else BLUE for v in s.values]
    ax.barh([label(i) for i in s.index], s.values, color=colors)
    ax.set_xlabel("Зниження ROC-AUC при перемішуванні ознаки")
    ax.set_title("Важливість ознак (permutation importance)")
    fig.tight_layout()
    return fig


def comparison(table):
    metrics = ["Precision", "Recall", "F1", "ROC-AUC"]
    x = np.arange(len(table))
    width = 0.2
    colors = ["#93c5fd", "#3b82f6", RED, "#1e3a8a"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for i, (m, c) in enumerate(zip(metrics, colors)):
        ax.bar(x + (i - 1.5) * width, table[m].values, width, label=m, color=c)
    short = [name.split(" (")[0] for name in table.index]
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=4, loc="lower right")
    ax.set_title("Порівняння алгоритмів класифікації")
    fig.tight_layout()
    return fig


def probability_gauge(probability, threshold):
    fig, ax = plt.subplots(figsize=(6, 0.9))
    ax.barh([0], [1], color="#e2e8f0", height=0.5)
    color = RED if probability >= threshold else ("#f59e0b" if probability >= 0.2 else "#16a34a")
    ax.barh([0], [probability], color=color, height=0.5)
    ax.axvline(threshold, color="black", lw=1, ls="--")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel("Ймовірність несправності (пунктир — поріг рішення)")
    fig.tight_layout()
    return fig
