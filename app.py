"""Застосунок для прогнозування несправностей обладнання у промислових системах.

Запуск:  streamlit run app.py
"""

import pandas as pd
import streamlit as st

from src import charts, config, data_loader, models, preprocessing

st.set_page_config(page_title="Прогнозування несправностей обладнання",
                   page_icon="⚙️", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 2rem;}
    div[data-testid="stMetric"] {background: #f1f5f9; border-radius: 10px; padding: 10px 14px;}
    .result-ok   {background:#dcfce7; border-left:6px solid #16a34a; padding:14px; border-radius:8px;}
    .result-warn {background:#fef3c7; border-left:6px solid #f59e0b; padding:14px; border-radius:8px;}
    .result-bad  {background:#fee2e2; border-left:6px solid #dc2626; padding:14px; border-radius:8px;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
#  Підготовка даних (кешується, щоб не повторювати обчислення)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Обробка даних...")
def prepare_data(path, mtime):
    raw = data_loader.read_raw(path)
    report = preprocessing.quality_report(raw)
    clean, log = preprocessing.clean_data(raw)
    data = preprocessing.add_features(clean)
    corr = preprocessing.correlation_matrix(data)
    selected, dropped = preprocessing.select_features(corr)
    return {"raw": raw, "report": report, "clean": clean, "log": log, "data": data,
            "corr": corr, "selected": selected, "dropped": dropped}


def get_data():
    if not data_loader.dataset_exists():
        return None
    return prepare_data(str(config.DATA_FILE), config.DATA_FILE.stat().st_mtime)


def label(col):
    return config.LABELS.get(col, col)


def fmt(x):
    return f"{x:.3f}".replace(".", ",")


if "result" not in st.session_state:
    st.session_state["result"] = models.load_model()

# --------------------------------------------------------------------------- #
#  Бічна панель навігації
# --------------------------------------------------------------------------- #
PAGES = ["🏠 Головна", "📥 Дані", "🧹 Якість та очищення", "🔗 Кореляційний аналіз",
         "🤖 Навчання моделі", "🔮 Прогноз несправності", "📊 Інтерпретація результатів"]

st.sidebar.title("⚙️ PredictMaint")
st.sidebar.caption("Інтелектуальний аналіз даних для прогнозування несправностей обладнання")
page = st.sidebar.radio("Розділи", PAGES)

bundle = get_data()
result = st.session_state["result"]

st.sidebar.divider()
st.sidebar.write("**Стан системи**")
st.sidebar.write(("✅ Дані завантажено" if bundle else "❌ Дані не завантажено"))
st.sidebar.write(("✅ Модель навчено" if result else "⏳ Модель не навчено"))
if result:
    st.sidebar.caption(f"Алгоритм: {result['algorithm']}")


def need_data():
    if bundle is None:
        st.warning("Спочатку завантажте набір даних на сторінці «Головна».")
        st.stop()


def need_model():
    if st.session_state["result"] is None:
        st.warning("Спочатку навчіть модель на сторінці «Навчання моделі».")
        st.stop()


# --------------------------------------------------------------------------- #
#  Головна
# --------------------------------------------------------------------------- #
if page == PAGES[0]:
    st.title("Прогнозування несправностей обладнання у промислових системах")
    st.write(
        "Система аналізує показники роботи технологічного обладнання (температури, швидкість "
        "обертання, крутний момент, знос інструмента) та за допомогою методів Data Mining "
        "прогнозує ймовірність виникнення несправності і її найімовірніший тип.")

    c1, c2, c3 = st.columns(3)
    c1.info("**1. Дані**\n\nЗавантаження, перевірка якості, очищення та кореляційний аналіз.")
    c2.info("**2. Модель**\n\nНавчання класифікатора, оцінювання якості, порівняння алгоритмів.")
    c3.info("**3. Прогноз**\n\nКласифікація нового обладнання, рекомендації та інтерпретація.")

    st.subheader("Набір даних")
    st.write(
        "Використовується відкритий набір **AI4I 2020 Predictive Maintenance Dataset** "
        f"([UCI Machine Learning Repository]({config.DATA_PAGE})): 10 000 записів, "
        "6 вхідних характеристик, мітка несправності та 5 типів несправностей.")

    if bundle:
        st.success(f"Набір даних знайдено: `{config.DATA_FILE.name}` "
                   f"({bundle['report']['rows']} записів).")
    else:
        st.warning("Набір даних ще не завантажено.")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⬇️ Завантажити набір даних з UCI", type="primary"):
            with st.spinner("Завантаження..."):
                ok, msg = data_loader.download_dataset()
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
                st.info(f"Можна завантажити архів вручну зі сторінки {config.DATA_PAGE}, "
                        "розпакувати та вибрати файл ai4i2020.csv праворуч.")
    with col_b:
        uploaded = st.file_uploader("Або виберіть файл ai4i2020.csv", type="csv")
        if uploaded is not None:
            try:
                data_loader.read_raw(uploaded)  # перевірка структури
                config.DATA_DIR.mkdir(parents=True, exist_ok=True)
                config.DATA_FILE.write_bytes(uploaded.getvalue())
                st.success("Файл збережено.")
                st.rerun()
            except Exception as exc:
                st.error(f"Некоректний файл: {exc}")

    st.subheader("Порядок роботи")
    st.markdown(
        "1. Перегляньте дані на сторінці **«Дані»**.\n"
        "2. Ознайомтеся з результатами перевірки на сторінці **«Якість та очищення»**.\n"
        "3. Проаналізуйте зв'язки між ознаками на сторінці **«Кореляційний аналіз»**.\n"
        "4. Навчіть модель на сторінці **«Навчання моделі»**.\n"
        "5. Введіть параметри обладнання на сторінці **«Прогноз несправності»**.\n"
        "6. Перегляньте висновки на сторінці **«Інтерпретація результатів»**.")

# --------------------------------------------------------------------------- #
#  Дані
# --------------------------------------------------------------------------- #
elif page == PAGES[1]:
    need_data()
    raw = bundle["raw"]
    st.title("Огляд вхідних даних")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Записів", f"{len(raw):,}".replace(",", " "))
    c2.metric("Характеристик", raw.shape[1])
    c3.metric("Випадків несправності", int(raw["failure"].sum()))
    c4.metric("Частка несправностей", f"{raw['failure'].mean() * 100:.2f} %")

    st.subheader("Перші записи набору даних")
    st.dataframe(raw.head(20))

    with st.expander("Опис характеристик"):
        st.markdown(
            "| Колонка | Опис |\n|---|---|\n"
            "| UDI, Product ID | Ідентифікатори запису та виробу |\n"
            "| Type | Клас якості продукції: L (50 %), M (30 %), H (20 %) |\n"
            "| Air temperature [K] | Температура навколишнього повітря |\n"
            "| Process temperature [K] | Температура технологічного процесу |\n"
            "| Rotational speed [rpm] | Швидкість обертання шпинделя |\n"
            "| Torque [Nm] | Крутний момент |\n"
            "| Tool wear [min] | Час роботи (знос) інструмента |\n"
            "| Machine failure | Цільова змінна: 1 — несправність, 0 — норма |\n"
            "| TWF, HDF, PWF, OSF, RNF | Тип несправності |")

    st.subheader("Описова статистика")
    stats = raw[config.NUMERIC_RAW].describe().T
    stats.index = [label(i) for i in stats.index]
    stats.columns = ["Кількість", "Середнє", "Ст. відхилення", "Мін", "25 %", "Медіана",
                     "75 %", "Макс"]
    st.dataframe(stats.round(2))

    c1, c2 = st.columns(2)
    c1.pyplot(charts.class_balance(raw))
    c2.pyplot(charts.failure_types(raw))
    st.info("Класи незбалансовані: несправності становлять лише кілька відсотків записів. "
            "Тому під час навчання використовується зважування класів, а якість моделі "
            "оцінюється не лише точністю (Accuracy), а й метриками Precision, Recall та F1.")

# --------------------------------------------------------------------------- #
#  Якість та очищення
# --------------------------------------------------------------------------- #
elif page == PAGES[2]:
    need_data()
    rep = bundle["report"]
    st.title("Оцінювання якості та очищення даних")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Пропущені значення", int(rep["missing"].sum()))
    c2.metric("Дублікати", rep["duplicates"] + rep["duplicate_ids"])
    c3.metric("Неможливі значення", int(rep["invalid"].sum()))
    c4.metric("Суперечливі записи", rep["failure_without_type"] + rep["type_without_failure"])

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Пропуски за колонками")
        st.dataframe(rep["missing"].rename("Кількість пропусків"))
    with c2:
        st.subheader("Викиди (правило 1,5·IQR)")
        out = rep["outliers"].rename("Кількість викидів")
        out.index = [label(i) for i in out.index]
        st.dataframe(out)
        st.caption("Викиди не видаляються: значення фізично можливі, а екстремальні режими "
                   "роботи якраз і пов'язані з несправностями.")

    st.pyplot(charts.boxplots(bundle["raw"], config.NUMERIC_RAW))

    st.subheader("Виконані дії з очищення")
    log = pd.DataFrame(bundle["log"], columns=["Дія", "Кількість записів"])
    st.dataframe(log, hide_index=True)
    st.write(
        f"Виявлено **{rep['failure_without_type']}** записів, де зафіксовано несправність без "
        f"зазначення її типу, та **{rep['type_without_failure']}** записів, де тип вказано, "
        "а несправність не зафіксовано. Такі записи суперечливі, тому їх вилучено.")

    st.subheader("Нові (похідні) ознаки")
    st.markdown(
        "- **Різниця температур** ΔT = T процесу − T повітря, K — характеризує тепловідведення;\n"
        "- **Механічна потужність** P = M · 2π · n / 60, Вт — характеризує навантаження на привід;\n"
        "- **Клас якості** L/M/H закодовано числами 0/1/2.")
    st.dataframe(bundle["data"].head(10))

    st.download_button("💾 Завантажити очищені дані (CSV)",
                       bundle["data"].to_csv(index=False).encode("utf-8"),
                       file_name="ai4i2020_clean.csv", mime="text/csv")

# --------------------------------------------------------------------------- #
#  Кореляційний аналіз
# --------------------------------------------------------------------------- #
elif page == PAGES[3]:
    need_data()
    data, corr = bundle["data"], bundle["corr"]
    st.title("Кореляційний аналіз та відбір ознак")
    st.write("Обчислено коефіцієнти кореляції Пірсона між усіма ознаками та цільовою змінною. "
             f"Сильною вважається кореляція з модулем не менше {config.CORR_THRESHOLD}.")

    st.pyplot(charts.correlation_heatmap(corr))

    pairs = preprocessing.strong_pairs(corr)
    st.subheader("Пари сильно корельованих ознак")
    if pairs:
        st.dataframe(pd.DataFrame([(label(a), label(b), r) for a, b, r in pairs],
                                  columns=["Ознака 1", "Ознака 2", "r"]), hide_index=True)
        cols = st.columns(min(len(pairs), 3))
        for col, (a, b, _) in zip(cols, pairs[:3]):
            col.pyplot(charts.scatter_pair(data, a, b))
    else:
        st.write("Сильно корельованих пар не виявлено.")

    st.subheader("Зв'язок ознак із несправністю")
    target_corr = corr[config.TARGET].drop(config.TARGET).sort_values(key=abs, ascending=False)
    st.dataframe(pd.DataFrame({"Ознака": [label(i) for i in target_corr.index],
                               "r з несправністю": target_corr.round(3).values}),
                 hide_index=True)

    st.subheader("Результат автоматичного відбору")
    st.write("Ознаки впорядковано за силою зв'язку з несправністю. З кожної пари сильно "
             "корельованих ознак залишається та, що сильніше пов'язана з цільовою змінною.")
    for feat, kept in bundle["dropped"].items():
        st.write(f"❌ **{label(feat)}** вилучено — сильна кореляція з ознакою «{label(kept)}» "
                 f"(r = {corr.loc[feat, kept]:.3f}).")
    st.success("Відібрані ознаки: " + ", ".join(label(f) for f in bundle["selected"]))

    st.pyplot(charts.distributions(data, preprocessing.CANDIDATE_FEATURES[1:]))

# --------------------------------------------------------------------------- #
#  Навчання моделі
# --------------------------------------------------------------------------- #
elif page == PAGES[4]:
    need_data()
    data = bundle["data"]
    st.title("Навчання моделі класифікації")

    with st.form("train"):
        c1, c2 = st.columns(2)
        algorithm = c1.selectbox("Алгоритм", list(models.ALGORITHMS),
                                 index=list(models.ALGORITHMS).index(models.DEFAULT_ALGORITHM))
        features = c2.multiselect("Ознаки для навчання", preprocessing.CANDIDATE_FEATURES,
                                  default=bundle["selected"], format_func=label)
        c3, c4, c5 = st.columns(3)
        test_size = c3.slider("Частка тестової вибірки", 0.1, 0.4, 0.25, 0.05)
        threshold = c4.slider("Поріг класифікації", 0.1, 0.9, 0.5, 0.05,
                              help="Якщо ймовірність несправності не менша за поріг, "
                                   "обладнання класифікується як несправне.")
        scale = c5.checkbox("Стандартизація ознак (z-оцінка)", value=True)
        submitted = st.form_submit_button("🚀 Навчити модель", type="primary")

    if submitted:
        if len(features) < 2:
            st.error("Виберіть щонайменше дві ознаки.")
            st.stop()
        with st.spinner("Навчання та крос-валідація моделі... (до хвилини)"):
            res = models.train_model(data, features, algorithm, test_size, scale, threshold)
            models.save_model(res)
        st.session_state["result"] = res
        st.success("Модель навчено та збережено у файлі models/failure_model.joblib.")

    res = st.session_state["result"]
    if res is not None:
        st.subheader(f"Результати: {res['algorithm']}")
        st.caption(f"Навчальна вибірка — {res['n_train']} записів, тестова — {res['n_test']}. "
                   f"Ознаки: {', '.join(label(f) for f in res['features'])}.")
        cols = st.columns(5)
        for col, (name, value) in zip(cols, res["metrics"].items()):
            col.metric(name, fmt(value))

        c1, c2 = st.columns(2)
        c1.pyplot(charts.confusion(res["confusion"]))
        c2.pyplot(charts.roc(*res["roc"], res["metrics"]["ROC-AUC"]))

        tn, fp, fn, tp = res["confusion"].ravel()
        st.write(f"З {tp + fn} несправностей тестової вибірки модель правильно виявила **{tp}**, "
                 f"пропустила **{fn}**. Хибних тривог — **{fp}** із {tn + fp} справних випадків.")

        st.subheader("5-кратна крос-валідація")
        cv = pd.DataFrame(res["cv"], index=["Середнє", "Ст. відхилення"]).T
        cv.index = [i.upper().replace("ROC_AUC", "ROC-AUC") for i in cv.index]
        st.dataframe(cv.round(3))

    st.divider()
    st.subheader("Порівняння алгоритмів")
    if st.button("📈 Порівняти всі алгоритми"):
        feats = res["features"] if res else bundle["selected"]
        with st.spinner("Навчання всіх алгоритмів..."):
            table = models.compare_algorithms(data, feats)
        st.session_state["comparison"] = table
    if "comparison" in st.session_state:
        table = st.session_state["comparison"]
        st.dataframe(table.round(3))
        st.pyplot(charts.comparison(table))
        st.write(f"Найкращий результат за F1-мірою показав алгоритм **{table.index[0]}**.")

# --------------------------------------------------------------------------- #
#  Прогноз
# --------------------------------------------------------------------------- #
elif page == PAGES[5]:
    need_data()
    need_model()
    res = st.session_state["result"]
    med = bundle["data"].median(numeric_only=True)
    st.title("Прогноз несправності для нового обладнання")

    tab1, tab2 = st.tabs(["Одна одиниця обладнання", "Пакетний прогноз (CSV)"])

    with tab1:
        with st.form("predict"):
            c1, c2, c3 = st.columns(3)
            ptype = c1.selectbox("Клас якості продукції", list(config.TYPE_NAMES),
                                 format_func=lambda t: config.TYPE_NAMES[t])
            air = c2.number_input("Температура повітря, K", 290.0, 310.0,
                                  round(float(med["air_temp"]), 1), 0.1)
            proc = c3.number_input("Температура процесу, K", 300.0, 320.0,
                                   round(float(med["process_temp"]), 1), 0.1)
            c4, c5, c6 = st.columns(3)
            speed = c4.number_input("Швидкість обертання, об/хв", 1000, 3000,
                                    int(med["rot_speed"]), 10)
            torque = c5.number_input("Крутний момент, Н·м", 1.0, 90.0,
                                     round(float(med["torque"]), 1), 0.5)
            wear = c6.number_input("Знос інструмента, хв", 0, 300, int(med["tool_wear"]), 1)
            go = st.form_submit_button("🔮 Спрогнозувати", type="primary")

        if go:
            row = pd.DataFrame([{"type": ptype, "air_temp": air, "process_temp": proc,
                                 "rot_speed": speed, "torque": torque, "tool_wear": wear}])
            row = preprocessing.add_features(row)
            pred = models.predict(res, row).iloc[0]
            p = pred["Ймовірність несправності"]

            css = "result-bad" if p >= res["threshold"] else ("result-warn" if p >= 0.2
                                                               else "result-ok")
            verdict = ("⚠️ Прогнозується НЕСПРАВНІСТЬ" if p >= res["threshold"]
                       else "✅ Обладнання у нормальному стані")
            st.markdown(f"<div class='{css}'><h3 style='margin:0'>{verdict}</h3>"
                        f"Ймовірність несправності: <b>{p * 100:.1f} %</b> · "
                        f"рівень ризику: <b>{pred['Рівень ризику']}</b></div>",
                        unsafe_allow_html=True)
            st.pyplot(charts.probability_gauge(p, res["threshold"]))

            c1, c2 = st.columns(2)
            c1.metric("Різниця температур, K", f"{row['temp_diff'].iloc[0]:.1f}")
            c2.metric("Механічна потужність, Вт", f"{row['power'].iloc[0]:.0f}")

            if p >= 0.2:
                st.subheader("Ймовірні типи несправності")
                tp = models.type_probabilities(res, row)
                st.dataframe(pd.DataFrame({
                    "Тип": [config.FAILURE_TYPES[t] for t in tp.index],
                    "Ймовірність, %": (tp.values * 100).round(1),
                    "Рекомендація": [config.RECOMMENDATIONS[t] for t in tp.index]}),
                    hide_index=True)

    with tab2:
        st.write("Завантажте CSV-файл з колонками: `type, air_temp, process_temp, rot_speed, "
                 "torque, tool_wear` (або з оригінальними назвами колонок набору AI4I).")
        template = pd.DataFrame([
            {"type": "L", "air_temp": 298.1, "process_temp": 308.6, "rot_speed": 1551,
             "torque": 42.8, "tool_wear": 0},
            {"type": "M", "air_temp": 302.5, "process_temp": 310.3, "rot_speed": 1320,
             "torque": 61.0, "tool_wear": 215},
            {"type": "H", "air_temp": 299.0, "process_temp": 309.8, "rot_speed": 2700,
             "torque": 12.5, "tool_wear": 120},
        ])
        st.download_button("Завантажити шаблон CSV", template.to_csv(index=False).encode(),
                           file_name="template.csv", mime="text/csv")
        batch = st.file_uploader("CSV-файл з параметрами обладнання", type="csv", key="batch")
        if batch is not None:
            try:
                new = pd.read_csv(batch).rename(columns=config.COLUMN_MAP)
                needed = ["type"] + config.NUMERIC_RAW
                missing = [c for c in needed if c not in new.columns]
                if missing:
                    st.error("Відсутні колонки: " + ", ".join(missing))
                else:
                    new = new[new["type"].isin(config.TYPE_CODES)].copy()
                    new = preprocessing.add_features(new[needed])
                    out = pd.concat([new[needed], models.predict(res, new)], axis=1)
                    st.dataframe(out)
                    n_bad = int((out["Прогноз"] == "Несправність").sum())
                    st.write(f"Прогнозується несправність для **{n_bad}** з {len(out)} одиниць.")
                    st.download_button("💾 Завантажити результати",
                                       out.to_csv(index=False).encode("utf-8"),
                                       file_name="predictions.csv", mime="text/csv")
            except Exception as exc:
                st.error(f"Помилка обробки файлу: {exc}")

# --------------------------------------------------------------------------- #
#  Інтерпретація
# --------------------------------------------------------------------------- #
elif page == PAGES[6]:
    need_data()
    need_model()
    res = st.session_state["result"]
    data = bundle["data"]
    m = res["metrics"]
    st.title("Інтерпретація результатів")

    st.subheader("Які ознаки найбільше впливають на прогноз")
    st.pyplot(charts.importance(res["importance"]))
    top = res["importance"].index[:3]
    st.write("Найбільший вплив на прогноз мають: " +
             ", ".join(f"**{label(f)}**" for f in top) +
             ". Перемішування значень цих ознак найсильніше погіршує якість моделі.")

    st.subheader("Оцінка якості прогнозу")
    st.markdown(
        f"- **Recall = {fmt(m['Recall'])}** — модель виявляє {m['Recall'] * 100:.0f} % "
        "реальних несправностей. Для промислового обладнання це найважливіша метрика, "
        "оскільки пропущена несправність призводить до аварійного простою.\n"
        f"- **Precision = {fmt(m['Precision'])}** — {m['Precision'] * 100:.0f} % сигналів "
        "тривоги є обґрунтованими; решта — хибні тривоги, що спричиняють зайві огляди.\n"
        f"- **F1 = {fmt(m['F1'])}** — збалансована оцінка precision і recall.\n"
        f"- **ROC-AUC = {fmt(m['ROC-AUC'])}** — здатність моделі розрізняти справний і "
        "несправний стан незалежно від порогу (1 — ідеально, 0,5 — випадкове вгадування).\n"
        f"- **Accuracy = {fmt(m['Accuracy'])}** — через незбалансованість класів ця метрика "
        "сама по собі малоінформативна.")

    st.subheader("Частота несправностей за інтервалами значень")
    feat = st.selectbox("Ознака", [f for f in preprocessing.CANDIDATE_FEATURES
                                   if f != "type_code"], format_func=label,
                        index=4)
    st.dataframe(models.failure_rate_by_bins(data, feat))

    st.subheader("Правила, виявлені деревом рішень (глибина 3)")
    st.caption("Неглибоке дерево рішень, побудоване на тих самих ознаках, дає зрозумілі "
               "людині правила виникнення несправностей.")
    st.code(models.extract_rules(data, res["features"]), language="text")

    st.subheader("Висновок")
    st.write(
        f"Модель «{res['algorithm']}» прогнозує несправності обладнання з F1-мірою "
        f"{fmt(m['F1'])} та ROC-AUC {fmt(m['ROC-AUC'])}. Основні чинники ризику — "
        f"{label(top[0]).lower()} та {label(top[1]).lower()}. Застосунок можна використовувати "
        "для планування технічного обслуговування за фактичним станом обладнання: вчасно "
        "замінювати інструмент, контролювати навантаження та охолодження.")
