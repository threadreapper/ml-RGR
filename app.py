import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ML Инференс CS:GO", layout="wide")

@st.cache_resource
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv('data/cs-ready.csv')

st.sidebar.title("Навигация")
page = st.sidebar.radio("Выберите страницу:", [
    "1. О разработчике",
    "2. О наборе данных",
    "3. Визуализации",
    "4. Инференс моделей"
])

if page == "1. О разработчике":
    st.header("Информация о разработчике")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.image("images/me.jpg", caption="вот он я")
    
    with col2:
        st.subheader("ФИО: Полушкин Тимофей Александрович")
        st.subheader("Группа: МО-241")
        st.markdown("---")
        st.write("**Тема РГР:** Разработка Web-приложения (дашборда) для инференса моделей ML и анализа данных на примере датасета CS:GO")
        st.write("**Цель:** Создание интерактивного дашборда для предсказания установки бомбы в раунде CS:GO с использованием 6 моделей машинного обучения.")

elif page == "2. О наборе данных":
    st.header("Информация о наборе данных CS:GO")
    
    st.markdown("""
    **Описание предметной области:**  
    Датасет содержит статистические данные раундов профессиональных матчей по игре Counter-Strike: Global Offensive. 
    Цель — предсказать, установлена ли бомба (bomb_planted), на основе текущей игровой ситуации.
    
    **Признаки:**
    - `time_left` — оставшееся время раунда (секунды)
    - `ct_score`, `t_score` — текущий счет команд
    - `map` — игровая карта (de_dust2, de_mirage и др.)
    - `ct_health`, `t_health` — суммарное здоровье команд (0-500)
    - `ct_armor`, `t_armor` — суммарная броня команд
    - `ct_money`, `t_money` — суммарные деньги команд ($)
    - `ct_helmets`, `t_helmets` — количество шлемов
    - `ct_defuse_kits` — количество наборов сапера
    - `ct_players_alive`, `t_players_alive` — живые игроки (0-5)
    
    **Целевой признак:**
    - `bomb_planted` — установлена ли бомба (0 - Нет, 1 - Да)
    
    **Предобработка данных:**
    1. Обработка пропущенных значений.
    2. Кодирование категориальных признаков (One-Hot Encoding для карт).
    3. Масштабирование числовых признаков (StandardScaler).
    """)
    
    df = load_data()
    st.subheader("Пример данных (первые 5 строк)")
    st.dataframe(df.head())

elif page == "3. Визуализации":
    st.header("Визуализации зависимостей в наборе данных")
    
    df = load_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Распределение здоровья команд")
        fig1, ax1 = plt.subplots()
        sns.histplot(df['ct_health'], color='blue', label='CT Health', kde=True, ax=ax1)
        sns.histplot(df['t_health'], color='red', label='T Health', kde=True, ax=ax1)
        ax1.legend()
        st.pyplot(fig1)
        
        st.subheader("2. Количество раундов по картам")
        fig2, ax2 = plt.subplots()
        sns.countplot(data=df, x='map', ax=ax2)
        ax2.tick_params(axis='x', rotation=45)
        st.pyplot(fig2)

    with col2:
        st.subheader("3. Зависимость денег от счета")
        fig3, ax3 = plt.subplots()
        sns.scatterplot(data=df.sample(500), x='ct_score', y='ct_money', alpha=0.6, ax=ax3)
        st.pyplot(fig3)
        
        st.subheader("4. Корреляционная матрица числовых признаков")
        numeric_df = df.select_dtypes(include=[np.number])
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', ax=ax4)
        st.pyplot(fig4)

elif page == "4. Инференс моделей":
    st.header("Получение предсказания модели ML")
    
    model_names = {
        "ML1: Логистическая регрессия": "logistic_regression.pkl",
        "ML2: Градиентный бустинг (Sklearn)": "gradient_boosting.pkl",
        "ML3: CatBoost": "catboost_model.pkl",
        "ML4: Бэггинг": "bagging_model.pkl",
        "ML5: Нейронная сеть (MLP)": "mlp_model.pkl",
        "ML6: Стэкинг": "stacking_model.pkl"
    }
    
    selected_model_name = st.selectbox("Выберите модель машинного обучения:", list(model_names.keys()))
    model_path = os.path.join('models', model_names[selected_model_name])
    
    model = load_model(model_path)
    
    st.markdown("---")
    st.subheader("Ввод данных для предсказания")
    
    input_method = st.radio("Способ ввода данных:", ["Ручной ввод", "Загрузка CSV файла"])
    
    if input_method == "Ручной ввод":
        with st.form("input_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                time_left = st.slider("Оставшееся время (сек)", 0.0, 175.0, 100.0)
                ct_score = st.number_input("Счет CT", min_value=0, value=0)
                t_score = st.number_input("Счет T", min_value=0, value=0)
                map_name = st.selectbox("Карта", ['de_dust2', 'de_mirage', 'de_nuke', 'de_inferno', 'de_overpass'])
                
            with col2:
                ct_health = st.slider("Здоровье CT (0-500)", 0, 500, 500)
                t_health = st.slider("Здоровье T (0-500)", 0, 500, 500)
                ct_armor = st.slider("Броня CT", 0, 500, 200)
                t_armor = st.slider("Броня T", 0, 500, 200)
                
            with col3:
                ct_money = st.number_input("Деньги CT ($)", min_value=0, value=4000)
                t_money = st.number_input("Деньги T ($)", min_value=0, value=4000)
                ct_helmets = st.number_input("Шлемы CT", min_value=0, max_value=5, value=0)
                t_helmets = st.number_input("Шлемы T", min_value=0, max_value=5, value=0)
                ct_defuse_kits = st.number_input("Наборы сапера CT", min_value=0, max_value=5, value=0)
                ct_players_alive = st.number_input("Живые CT", min_value=0, max_value=5, value=5)
                t_players_alive = st.number_input("Живые T", min_value=0, max_value=5, value=5)
                
            submit_button = st.form_submit_button("Получить предсказание")
            
            if submit_button:
                input_dict = {
                    'time_left': [time_left],
                    'ct_score': [ct_score],
                    't_score': [t_score],
                    'map': [map_name],
                    'ct_health': [ct_health],
                    't_health': [t_health],
                    'ct_armor': [ct_armor],
                    't_armor': [t_armor],
                    'ct_money': [ct_money],
                    't_money': [t_money],
                    'ct_helmets': [ct_helmets],
                    't_helmets': [t_helmets],
                    'ct_defuse_kits': [ct_defuse_kits],
                    'ct_players_alive': [ct_players_alive],
                    't_players_alive': [t_players_alive]
                }
                input_df = pd.DataFrame(input_dict)
                
                prediction = model.predict(input_df)[0]
                class_name = "Бомба установлена" if prediction == 1 else "Бомба НЕ установлена"
                
                st.success(f"Результат предсказания: **{class_name}**")
    
    elif input_method == "Загрузка CSV файла":
        st.info("Загрузите файл в формате *.csv с признаками раундов (без колонки bomb_planted)")
        uploaded_file = st.file_uploader("Выберите CSV файл", type="csv")
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("Предпросмотр загруженных данных:", data.head())
            
            try:
                predictions = model.predict(data)
                data['Prediction'] = predictions
                data['Prediction'] = data['Prediction'].map({0: 'Not Planted', 1: 'Bomb Planted'})
                
                st.success("Предсказания успешно получены!")
                st.dataframe(data)
                
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Скачать результаты в CSV",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv'
                )
            except Exception as e:
                st.error(f"Ошибка при предсказании: {e}. Проверьте формат данных в файле.")
