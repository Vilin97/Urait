# app.py
import streamlit as st
import pandas as pd
import src.utils as utils

st.set_page_config(page_title="Подбор курса по дисциплине", layout="centered")

YES_TOKENS = {"да", "yes", "true", "1"}
NO_TOKENS  = {"нет", "no", "false", "0"}

def to_bool_ru(x: object) -> bool:
    s = str(x).strip().lower()
    if s in YES_TOKENS: return True
    if s in NO_TOKENS:  return False
    return False  # по умолчанию — «нет»

DEFAULT_PROMPT = (
    "Извлеки название дисциплины/курса и названия тем, которые в нём изучаются, "
    "всё на русском языке. Включай только академические темы, не административные. "
    "Ответь только названиями, разделёнными запятыми. Название дисциплины/курса должно быть первым."
)

@st.cache_resource
def get_client():
    return utils.get_api_client()

@st.cache_resource
def load_courses_and_embeddings():
    courses_df = pd.read_csv("courses.csv")
    embeddings_by_id = utils.load_course_embeddings()
    return courses_df, embeddings_by_id

def run_matching(url: str, parse_prompt: str, top_k: int = 5):
    client = get_client()

    # Шаг 1: парсинг
    st.info("🔍 Парсим документ…")
    parsed_topics = utils.parse_document(url, parse_prompt, client)
    toks = [s.strip() for s in parsed_topics.split(",") if s.strip()]
    if not toks:
        raise ValueError("Парсинг вернул пустой результат.")
    discipline, topics = toks[0], toks[1:]
    st.success("✅ Парсинг завершён.")
    with st.container(border=True):
        st.markdown("**Результаты парсинга:**")
        st.markdown(f"• **Дисциплина:** {discipline}")
        if topics:
            st.markdown("• **Темы:** " + ", ".join(topics))

    # Подготовка эмбеддинга
    text_to_embed = f"{discipline}, {', '.join(topics)}"
    embedding = utils.embed_text(text_to_embed, client)

    # Шаг 2: поиск ближайших курсов
    courses_df, embeddings_by_id = load_courses_and_embeddings()
    st.info(f"📊 Находим топ-{top_k} ближайших курсов…")
    most_similar = utils.get_most_similar(embedding, embeddings_by_id, top_k=top_k)  # [(id, score)]
    id2score = dict(most_similar)
    top_ids = [cid for cid, _ in most_similar]
    st.success(f"✅ Топ-{top_k} найден.")
    with st.container(border=True):
        st.markdown("**Найденные ближайшие курсы (ID → сходство):**")
        st.write([{"ID курса": cid, "Сходство": float(f"{id2score[cid]:.4f}")} for cid in top_ids])

    # Таблица кандидатов (пока без пригодности)
    sim_df = (
        courses_df[courses_df["project_id"].isin(top_ids)].copy()
        .assign(similarity=lambda df: df["project_id"].map(id2score).astype(float))
        .sort_values("similarity", ascending=False)
        .reset_index(drop=True)
    )

    # Шаг 3: проверка пригодности — с поэтапным выводом и досрочной остановкой
    st.info("🤖 Проверяем пригодность курсов…")
    log_box = st.container()  # сюда будем писать результаты по одному
    processed_rows = []
    early_stop = False

    for i, row in sim_df.iterrows():
        course_id = row["project_id"]
        course_name = row["project_name"]
        course_topics = row["topics"]

        with log_box.expander(f"Проверка #{i+1}: {course_name} (ID {course_id})", expanded=True if i == 0 else False):
            st.caption(f"Темы курса: {course_topics}")
            raw_decision, explanation = utils.determine_course_suitability(
                discipline, topics, course_name, course_topics, client
            )
            suitable = to_bool_ru(raw_decision)
            st.markdown(f"**Решение:** {'✅ Да' if suitable else '❌ Нет'}")
            st.markdown(f"**Пояснение:** {explanation}")

        processed_rows.append(
            {
                "project_id": course_id,
                "project_name": course_name,
                "topics": course_topics,
                "similarity": float(row["similarity"]),
                "Пригоден": bool(suitable),
                "Пояснение": explanation,
            }
        )

        if suitable:
            st.success("🛑 Найден подходящий курс — останавливаем проверку остальных.")
            early_stop = True
            break

    st.success("✅ Проверка пригодности завершена." if not early_stop else "✅ Проверка завершена досрочно.")

    # Собираем финальный DataFrame только из обработанных строк
    out_df = pd.DataFrame(processed_rows)
    out_df = out_df.rename(columns={
        "project_id":   "ID курса",
        "project_name": "Название курса",
        "similarity":   "Сходство",
        "topics":       "Темы курса",
    })
    out_df = out_df[["ID курса", "Название курса", "Темы курса", "Сходство", "Пригоден", "Пояснение"]]
    # Отсортируем по сходству (на случай досрочной остановки не по первому)
    out_df = out_df.sort_values("Сходство", ascending=False).reset_index(drop=True)

    return discipline, topics, out_df

# ===== UI =====
st.title("Подбор курса по дисциплине")
url = st.text_input("Введите URL РПД (PDF или HTML)")

col1, col2 = st.columns([1, 1])
with col1:
    top_k = st.number_input("Сколько курсов искать (top-k)?", min_value=1, max_value=50, value=5, step=1)
with col2:
    with st.expander("Промпт для парсинга (опционально)"):
        parse_prompt = st.text_area("Промпт", value=DEFAULT_PROMPT, height=120)

if st.button("Найти подходящие курсы") and url:
    with st.spinner("Идёт парсинг, векторизация и поиск…"):
        try:
            discipline, topics, sim_df = run_matching(url, parse_prompt, top_k=top_k)
        except Exception as e:
            st.error(f"Ошибка: {e}")
        else:
            st.subheader("Распознанная дисциплина")
            st.markdown(f"**Дисциплина:** {discipline}")
            if topics:
                st.markdown("**Темы:** " + ", ".join(topics))

            st.subheader(f"Проверенные кандидаты (до {top_k})")
            st.dataframe(
                sim_df,  # колонки: ID курса, Название курса, Темы курса, Сходство, Пригоден, Пояснение
                use_container_width=True,
            )

            # Итоговый вердикт
            yes = sim_df[sim_df["Пригоден"]]
            st.subheader("Вердикт")
            if len(yes):
                best = yes.sort_values("Сходство", ascending=False).iloc[0]
                st.success(f"✅ Пригоден: {best['Название курса']} (оценка сходства {best['Сходство']:.3f})")
                st.caption(best["Пояснение"])
            else:
                st.warning("❌ Подходящих курсов среди проверенных не найдено.")
