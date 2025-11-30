import streamlit as st
import numpy as np
import time
import pickle
import pandas as pd
import shap
import requests
import json
import os

# Перевод категориальных значений
GENDER_TRANSLATIONS = {
    "Male": "Мужской",
    "Female": "Женский",
    "Other": "Другой"
}
QUALITY_TRANSLATIONS = {
    "Excellent": "Отличное",
    "Good": "Хорошее",
    "Fair": "Удовлетворительное",
    "Poor": "Плохое"
}
STRESS_TRANSLATIONS = {
    "Low": "Низкий",
    "Medium": "Средний",
    "High": "Высокий"
}

# Обратные переводы для кодирования
GENDER_REVERSE_TRANSLATIONS = {v: k for k, v in GENDER_TRANSLATIONS.items()}
QUALITY_REVERSE_TRANSLATIONS = {v: k for k, v in QUALITY_TRANSLATIONS.items()}
STRESS_REVERSE_TRANSLATIONS = {v: k for k, v in STRESS_TRANSLATIONS.items()}

# === НАСТРОЙКИ ===
st.set_page_config(
    page_title="NeuroSleep – ИИ предсказатель снов",
    page_icon="brain",
    layout="centered"
)

# Helper function to convert SHAP value to text description
def get_impact_description(shap_value):
    direction = "положительное" if shap_value > 0 else "отрицательное"
    magnitude_val = abs(shap_value)
    
    if magnitude_val > 1.5:
        magnitude = "очень сильное"
    elif magnitude_val > 0.5:
        magnitude = "значительное"
    elif magnitude_val > 0.1:
        magnitude = "заметное"
    else:
        magnitude = "небольшое"
        
    return f"{magnitude} {direction} влияние"

# Function to get explanation from OpenRouter AI
def get_openrouter_explanation(shap_values_obj, input_df, predicted_type, target_encoder):
    OPENROUTER_API_KEY = "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"
    YOUR_SITE_URL = "http://localhost:8501" # Placeholder
    YOUR_SITE_NAME = "NeuroSleep"

    if not OPENROUTER_API_KEY:
        return "Ошибка: Ключ API OpenRouter не найден. Пожалуйста, убедитесь, что он задан в вашем .env файле."

    predicted_class_index = list(target_encoder.classes_).index(predicted_type)
    shap_values_for_class = shap_values_obj.values[0, :, predicted_class_index]
    
    prompt_intro = f"Ты — дружелюбный эксперт по сну по имени NeuroSleep. Твоя задача — объяснить пользователю простым языком, почему ему был предсказан тип сна '{predicted_type}'. Основывайся на данных, которые я тебе предоставлю. В твоем ответе не должно быть никаких чисел или числовых значений. Объясни 2-3 самых важных фактора и дай один конкретный, полезный совет для улучшения сна, связанный с самым негативным фактором. Ответ должен быть строго на русском языке, на других языках нельзя никак."
    
    factors_text = "Вот данные пользователя и как они повлияли на прогноз:\n"
    
    user_values = {
        'Age': input_df['Age'].iloc[0], 'Caffeine_mg': f"{input_df['Caffeine_mg'].iloc[0]} мг",
        'Sleep_Hours': f"{input_df['Sleep_Hours'].iloc[0]} часов", 'Physical_Activity_Hours': f"{input_df['Physical_Activity_Hours'].iloc[0]} часов",
        'Smoking': "Да" if input_df['Smoking'].iloc[0] == 1 else "Нет",
        'Alcohol_Consumption': "Да" if input_df['Alcohol_Consumption'].iloc[0] == 1 else "Нет",
        'Phone_Usage_Hours_Before_Sleep': f"{input_df['Phone_Usage_Hours_Before_Sleep'].iloc[0]} часов",
        'Gender': gender_encoder.inverse_transform([input_df['Gender_encoded'].iloc[0]])[0],
        'Sleep_Quality': quality_encoder.inverse_transform([input_df['Sleep_Quality_encoded'].iloc[0]])[0],
        'Stress_Level': stress_encoder.inverse_transform([input_df['Stress_Level_encoded'].iloc[0]])[0]
    }

    impacts = []
    for i, col in enumerate(input_df.columns):
        feature_name = col.replace('_encoded', '').replace('_', ' ').capitalize()
        shap_value = shap_values_for_class[i]
        user_value = user_values.get(col, user_values.get(feature_name, 'N/A'))
        impact_desc = get_impact_description(shap_value)
        impacts.append({'name': feature_name, 'value': user_value, 'desc': impact_desc, 'shap': shap_value})

    # Sort by absolute shap value to discuss most important factors
    impacts.sort(key=lambda x: abs(x['shap']), reverse=True)
    
    for impact in impacts:
        factors_text += f"- {impact['name']}: {impact['value']} (влияние: {impact['desc']})\n"

    full_prompt = f"{prompt_intro}\n\n{factors_text}"

    # Real API Call
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": YOUR_SITE_URL, "X-Title": YOUR_SITE_NAME,
                "Content-Type": "application/json"
            },
            data=json.dumps({
                "model": "x-ai/grok-4.1-fast:free",
                "messages": [{"role": "user", "content": full_prompt}]
            })
        )
        response.raise_for_status()
        ai_response = response.json()['choices'][0]['message']['content']
        return ai_response
    except requests.exceptions.RequestException as e:
        return f"### Не удалось получить объяснение от ИИ\n\nПроизошла ошибка при обращении к OpenRouter: `{e}`"
    except (KeyError, IndexError):
        return "### Не удалось получить объяснение от ИИ\n\nПолучен неожиданный ответ от сервера."


# Function to load the model and encoders
@st.cache_resource
def load_resources():
    try:
        model = pickle.load(open('sleep_type_model.pkl', 'rb'))
        gender_encoder = pickle.load(open('gender_encoder.pkl', 'rb'))
        target_encoder = pickle.load(open('target_encoder.pkl', 'rb'))
        quality_encoder = pickle.load(open('quality_encoder.pkl', 'rb'))
        stress_encoder = pickle.load(open('stress_encoder.pkl', 'rb'))
        shap_explainer = pickle.load(open('shap_explainer.pkl', 'rb'))
        return model, gender_encoder, target_encoder, quality_encoder, stress_encoder, shap_explainer
    except FileNotFoundError:
        return None, None, None, None, None, None # Return Nones if files not found

model, gender_encoder, target_encoder, quality_encoder, stress_encoder, shap_explainer = load_resources()

if model is None:
    st.error("Ошибка загрузки модели или кодировщиков. Убедитесь, что 'train_model.py' был запущен успешно и создал все .pkl файлы.")
    st.stop()
    
# === ДИЗАЙН ===
st.markdown("""
<style>
    /* .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        background-attachment: fixed;
    } */ /* Убрано для поддержки тем */

    h2, h3 {color: var(--text-color) !important; text-align: center;}

    .stButton > button {
        background: linear-gradient(45deg, #a855f7, #ec4899);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 18px 50px;
        font-size: 1.6rem;
        font-weight: bold;
        box-shadow: 0 15px 35px rgba(168, 85, 247, 0.6);
    }

    .stTextInput > div > div > input {
        background: var(--secondary-background-color) !important;
        border: 2px solid rgba(168, 85, 247, 0.6) !important;
        border-radius: 18px !important;
        color: var(--text-color) !important;
        padding: 18px 20px !important;
        font-size: 1.3rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #ec4899 !important;
        box-shadow: 0 0 20px rgba(236, 72, 153, 0.6) !important;
    }
    .stTextInput > label {color: var(--text-color) !important; font-size: 1.2rem; font-weight: bold; opacity: 0.8;}
</style>
""", unsafe_allow_html=True)

# === ЗВЁЗДЫ ===
st.markdown("""
<div style="position:fixed; top:0; left:0; width:100%; height:100%; pointer-events:none; z-index:-1; opacity:0.2;">
    <script>
        for(let i=0; i<300; i++){
            let s = document.createElement('div');
            s.style.position = 'absolute';
            s.style.width = s.style.height = Math.random()*4 + 'px';
            s.style.background = 'white';
            s.style.borderRadius = '50%';
            s.style.top = Math.random()*100 + '%';
            s.style.left = Math.random()*100 + '%';
            s.style.animation = 'twinkle ' + (Math.random()*5+5) + 's infinite';
            document.body.appendChild(s);
        }
    </script>
    <style>@keyframes twinkle{0%,100%{opacity:0.1}50%{opacity:1}}</style>
</div>
""", unsafe_allow_html=True)

# === ЗАГОЛОВОК ===
st.markdown("""
<h1 style="
    font-size: 5.5rem;
    color: #a855f7; /* Changed to a solid color for theme compatibility */
    text-align: center;
    font-weight: 900;
    margin: 20px 0 10px 0;
">NeuroSleep</h1>
""", unsafe_allow_html=True)
st.markdown("<h2>ИИ предсказывает твой сон до того, как ты его увидишь</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:var(--text-color); opacity: 0.7; font-size:1.5rem; margin-bottom:40px;'>Просто расскажи, как прошёл день</p>", unsafe_allow_html=True)

# === ПОЛЯ ВВОДА (без ограничений) ===
empty_col_left, main_input_col, empty_col_right = st.columns([1, 2, 1])

with main_input_col:
    # Existing inputs
    age_input = st.text_input("Возраст", placeholder="Сколько лет?")
    gender_input = st.selectbox("Пол", options=list(GENDER_TRANSLATIONS.values()))
    sleep_duration_input = st.text_input("Продолжительность сна (часов)", placeholder="Например, 7.5")
    physical_activity_input = st.text_input("Физическая активность (часов в день)", placeholder="Например, 2.5")
    
    # Updated/New inputs
    quality_input = st.selectbox("Качество сна", options=list(QUALITY_TRANSLATIONS.values()))
    stress_input = st.selectbox("Уровень стресса", options=list(STRESS_TRANSLATIONS.values()))
    caffeine_cups_input = st.text_input("Количество чашек кофе", placeholder="Например, 2")
    smoking_input = st.selectbox("Курение", ["Нет", "Да"])
    alcohol_input = st.selectbox("Алкоголь", ["Нет", "Да"])
    phone_usage_hours_before_sleep_input = st.text_input("Часов до сна использовали гаджеты", placeholder="Например, 1.5")
    
st.markdown("<br>", unsafe_allow_html=True)

# === ПРЕДСКАЗАНИЕ ===
empty_col_left_btn, button_col, empty_col_right_btn = st.columns([1, 1, 1])
with button_col:
    button_pressed = st.button("ПРЕДСКАЗАТЬ МОЙ СОН", type="primary")

if button_pressed:
    with st.spinner("NeuroSleep анализирует ваши данные..."):
        time.sleep(2) # Shorter sleep for better UX

    st.balloons()

    # Preprocess inputs
    try:
        # Convert all inputs to their correct types
        age = int(age_input)
        sleep_hours = float(sleep_duration_input)
        physical_activity_hours = float(physical_activity_input)
        phone_usage_hours_before_sleep = float(phone_usage_hours_before_sleep_input)
        
        # Convert coffee cups to caffeine_mg
        coffee_cups = float(caffeine_cups_input)
        caffeine_mg = coffee_cups * 85 # 1 cup = 85 mg caffeine
        
        smoking = 1 if smoking_input == "Да" else 0
        alcohol = 1 if alcohol_input == "Да" else 0

        # Encode categorical features
        gender_encoded = gender_encoder.transform([GENDER_REVERSE_TRANSLATIONS[gender_input]])[0]
        quality_encoded = quality_encoder.transform([QUALITY_REVERSE_TRANSLATIONS[quality_input]])[0]
        stress_encoded = stress_encoder.transform([STRESS_REVERSE_TRANSLATIONS[stress_input]])[0]

        # Create DataFrame for prediction with the correct column order
        input_data = pd.DataFrame([[ 
            age,
            caffeine_mg,
            sleep_hours,
            physical_activity_hours,
            smoking,
            alcohol,
            phone_usage_hours_before_sleep,
            gender_encoded,
            quality_encoded,
            stress_encoded
        ]], columns=[
            'Age', 'Caffeine_mg', 'Sleep_Hours', 'Physical_Activity_Hours',
            'Smoking', 'Alcohol_Consumption', 'Phone_Usage_Hours_Before_Sleep',
            'Gender_encoded', 'Sleep_Quality_encoded', 'Stress_Level_encoded'
        ])

        # Make prediction and decode it
        prediction_encoded = model.predict(input_data)[0]
        predicted_sleep_type = target_encoder.inverse_transform([prediction_encoded])[0]

        # Map prediction to Russian dreams
        sleep_type_mapping = {
            'Восстановительный': "Восстановительный — идеальный отдых, вы полны энергии!",
            'Спокойный': "Спокойный — отличный сон, который заряжает бодростью.",
            'Нормальный': "Нормальный — хороший сон, но можно стремиться к лучшему.",
            'Эмоциональный': "Эмоциональный — ваш мозг активно обрабатывал события дня.",
            'Прерывистый': "Прерывистый — возможно, что-то мешало вам спать. Стоит обратить внимание на вечерние ритуалы.",
            'Беспокойный': "Беспокойный — ваш сон был неглубоким. Попробуйте снизить стресс и расслабиться перед сном."
        }
        result = sleep_type_mapping.get(predicted_sleep_type, "Неизвестный тип сна")

    except ValueError:
        st.error("Пожалуйста, введите корректные числовые значения для всех полей.")
        result = "Ошибка ввода"
    except Exception as e:
        st.error(f"Произошла ошибка при предсказании: {e}")
        result = "Ошибка предсказания"

    if ' — ' in result:
        st.markdown(f"""
        <div style="text-align:center; padding:60px; background: linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899);
                    border-radius:35px; margin:50px 0; box-shadow: 0 30px 70px rgba(139, 92, 246, 0.7);">
            <h1 style="color:white; margin:0; font-size:3rem; word-break: break-word;">{result.split(' — ')[0]}</h1>
            <p style="color:#ddd6fe; font-size:1.5rem; margin:15px 0;">{result.split(' — ')[1]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # === SHAP Explanation ===
        with st.spinner("ИИ-ассистент анализирует ваш сон..."):
            # Calculate SHAP values for the single prediction
            shap_values_single = shap_explainer(input_data)
            
            # Get explanation from the AI
            ai_explanation = get_openrouter_explanation(
                shap_values_obj=shap_values_single,
                input_df=input_data,
                predicted_type=predicted_sleep_type,
                target_encoder=target_encoder
            )
            st.info(ai_explanation)

# === ФУТЕР ===
st.markdown("""
<div style='text-align:center; color:#c4b5fd; padding:40px; margin-top:60px;'>
    <h2>NeuroSleep</h2>
    <p><strong>Nexora Команда Genesis</strong></p>
    <p>Мы делаем сон будущего — уже сегодня</p>
</div>
""", unsafe_allow_html=True)
