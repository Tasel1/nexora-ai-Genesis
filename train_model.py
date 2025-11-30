# train_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
import xgboost as xgb
import shap

print("🚀 Начинаем загрузку и обработку данных...")

# Шаг 1: Проверяем наличие файла
file_path = "data/synthetic_coffee_health_10000.csv"

if not os.path.exists(file_path):
    print(f"❌ ОШИБКА: Файл {file_path} не найден!")
    print("Убедитесь, что:")
    print("1. Файл находится в папке 'data'")
    print("2. Название файла точно: 'synthetic_coffee_health_10000.csv'")
    exit()
else:
    print("✅ Файл найден!")

# Шаг 2: Загружаем данные
try:
    df = pd.read_csv(file_path)
    print("✅ Данные успешно загружены!")
    print(f"Размер датасета: {df.shape}")
except Exception as e:
    print(f"❌ Ошибка при загрузке: {e}")
    exit()

# Шаг 3: Просмотр структуры данных
print("\n📊 Первые 5 строк данных:")
print(df.head())

print("\n🔍 Информация о данных:")
print(df.info())

print("\n📈 Статистика данных:")
print(df.describe())

print("\n🎯 Столбцы в датасете:")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

print("\n🔢 Проверка пропущенных значений:")
print(df.isnull().sum())

print("\n🎯 Начинаем обработку данных для нашей задачи...")

# Функция для создания типов сна (обновленная с большим количеством классификаций)
def create_sleep_type(row):
    """
    Создаем более детализированные категории сна на основе системы оценки.
    """
    score = 0

    # Качество и продолжительность сна
    if row['Sleep_Quality'] == 'Excellent':
        score += 3
    elif row['Sleep_Quality'] == 'Good':
        score += 1
    elif row['Sleep_Quality'] == 'Fair':
        score -= 1
    elif row['Sleep_Quality'] == 'Poor':
        score -= 3
    
    if row['Sleep_Hours'] > 7.5:
        score += 2
    elif row['Sleep_Hours'] < 6:
        score -= 2

    # Уровень стресса
    if row['Stress_Level'] == 'Low':
        score += 1
    elif row['Stress_Level'] == 'Medium':
        score -= 1
    elif row['Stress_Level'] == 'High':
        score -= 3

    # Физическая активность
    if row['Physical_Activity_Hours'] > 5:
        score += 1
    elif row['Physical_Activity_Hours'] < 1:
        score -= 1
        
    # Вредные привычки
    if row['Alcohol_Consumption'] > 0:
        score -= 2
    if row['Smoking'] > 0:
        score -= 2

    # Кофеин
    if row['Caffeine_mg'] > 300:
        score -= 2
        
    # Возраст
    if row['Age'] > 60:
        score -= 1
        
    # Использование телефона перед сном
    if row['Phone_Usage_Hours_Before_Sleep'] < 1:
        score -= 3
    elif row['Phone_Usage_Hours_Before_Sleep'] < 2:
        score -= 1

    # Финальная классификация по баллам
    if score >= 4:
        return 'Восстановительный'
    elif score >= 2:
        return 'Спокойный'
    elif score >= 0:
        return 'Нормальный'
    elif score >= -2:
        return 'Эмоциональный'
    elif score >= -4:
        return 'Прерывистый'
    else:
        return 'Беспокойный'


# Симулируем данные об использовании телефона перед сном
def simulate_phone_usage(row):
    stress = row['Stress_Level']
    quality = row['Sleep_Quality']
    
    if stress == 'High' or quality == 'Poor':
        # Высокий стресс / плохое качество сна -> использование телефона ближе ко сну
        return round(np.random.uniform(0, 1.5), 1)
    elif stress == 'Medium' or quality == 'Fair':
        return round(np.random.uniform(0.5, 3.0), 1)
    else: # Низкий стресс / хорошее или отличное качество
        return round(np.random.uniform(1.0, 4.0), 1)

df['Phone_Usage_Hours_Before_Sleep'] = df.apply(simulate_phone_usage, axis=1)

# Применяем функцию для создания целевой переменной
df['sleep_type'] = df.apply(create_sleep_type, axis=1)

print("✅ Целевая переменная 'sleep_type' создана!")
print("✅ Симулированы данные об использовании телефона!")
print("\n📊 Распределение типов сна:")
print(df['sleep_type'].value_counts())

# Визуализация распределения
plt.figure(figsize=(10, 6))
df['sleep_type'].value_counts().plot(kind='bar', color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
plt.title('Распределение типов сна в датасете')
plt.xlabel('Тип сна')
plt.ylabel('Количество')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('sleep_type_distribution.png')
# plt.show() # Removed to prevent blocking

print("📊 Визуализация сохранена как 'sleep_type_distribution.png'")

# Продолжение train_model.py

print("\n🔧 Подготовка признаков для модели...")

# Выбираем только нужные признаки
selected_features = ['Age', 'Gender', 'Caffeine_mg', 
                    'Sleep_Hours', 'Sleep_Quality', 'Stress_Level', 'Physical_Activity_Hours',
                    'Smoking', 'Alcohol_Consumption', 'Phone_Usage_Hours_Before_Sleep']

# Создаем финальный DataFrame с выбранными признаками
features_df = df[selected_features].copy()

# Кодируем категориальные переменные
from sklearn.preprocessing import LabelEncoder

# Кодируем пол (если есть 'Other', его тоже кодируем)
gender_encoder = LabelEncoder()
features_df['Gender_encoded'] = gender_encoder.fit_transform(features_df['Gender'])

# Кодируем качество сна
quality_encoder = LabelEncoder()
features_df['Sleep_Quality_encoded'] = quality_encoder.fit_transform(features_df['Sleep_Quality'])

# Кодируем уровень стресса
stress_encoder = LabelEncoder()
features_df['Stress_Level_encoded'] = stress_encoder.fit_transform(features_df['Stress_Level'])

# Удаляем исходные категориальные столбцы
features_df = features_df.drop(['Gender', 'Sleep_Quality', 'Stress_Level'], axis=1)

print("✅ Признаки подготовлены!")
print("\n📋 Итоговые признаки для модели:")
print(features_df.columns.tolist())

# Подготовка данных для обучения
X = features_df
y = df['sleep_type']

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

print(f"\n📊 Размеры данных:")
print(f"Признаки (X): {X.shape}")
print(f"Целевая переменная (y_encoded): {y_encoded.shape}")

# Разделяем на обучающую и тестовую выборки
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\n📚 Разделение данных:")
print(f"Обучающая выборка: {X_train.shape}")
print(f"Тестовая выборка: {X_test.shape}")

# Шаг 7: Обучение модели
print("\n🤖 Обучаем модель XGBoost...")

model = xgb.XGBClassifier(
    objective='multi:softmax', # For multiclass classification, outputs class labels
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    use_label_encoder=False, # Suppress deprecation warning
    eval_metric='mlogloss', # Metric for multiclass logloss
    random_state=42
)

model.fit(X_train, y_train_encoded)

# Предсказания на тестовой выборке
y_pred_encoded = model.predict(X_test)

# Оценка модели
accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
print(f"✅ Точность модели: {accuracy:.2f}")

print("\n📊 Детальный отчет по классификации:")
print(classification_report(y_test_encoded, y_pred_encoded, target_names=target_encoder.classes_))

# Шаг 8: Сохранение модели
print("\n💾 Сохраняем модель...")

# Сохраняем модель
with open('sleep_type_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Сохраняем кодировщик пола
with open('gender_encoder.pkl', 'wb') as f:
    pickle.dump(gender_encoder, f)

# Сохраняем кодировщик целевой переменной
with open('target_encoder.pkl', 'wb') as f:
    pickle.dump(target_encoder, f)

# Сохраняем кодировщик качества сна
with open('quality_encoder.pkl', 'wb') as f:
    pickle.dump(quality_encoder, f)

# Сохраняем кодировщик уровня стресса
with open('stress_encoder.pkl', 'wb') as f:
    pickle.dump(stress_encoder, f)

print("✅ Модель сохранена как 'sleep_type_model.pkl'")
print("✅ Кодировщик пола сохранен как 'gender_encoder.pkl'")
print("✅ Кодировщик целевой переменной сохранен как 'target_encoder.pkl'")
print("✅ Кодировщик качества сна сохранен как 'quality_encoder.pkl'")
print("✅ Кодировщик уровня стресса сохранен как 'stress_encoder.pkl'")

# Шаг 9: Анализ важности признаков с помощью SHAP
print("\n📈 Анализ важности признаков с помощью SHAP...")

# Создаем объяснитель SHAP
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Глобальная важность признаков
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title('Глобальная важность признаков (SHAP)')
plt.tight_layout()
plt.savefig('shap_summary.png')
# plt.show() # Убираем, чтобы не блокировать выполнение

print("✅ График важности признаков SHAP сохранен как 'shap_summary.png'")

# Сохраняем объяснитель SHAP
with open('shap_explainer.pkl', 'wb') as f:
    pickle.dump(explainer, f)
    
print("✅ Объяснитель SHAP сохранен как 'shap_explainer.pkl'")


print("🎉 Обучение модели завершено!")
print("\n📁 Созданные файлы:")
print("1. sleep_type_model.pkl - обученная модель")
print("2. gender_encoder.pkl, quality_encoder.pkl, stress_encoder.pkl - кодировщики")
print("3. target_encoder.pkl - кодировщик целевой переменной")
print("4. shap_explainer.pkl - объяснитель SHAP")
print("5. sleep_type_distribution.png - распределение типов сна")
print("6. shap_summary.png - график важности признаков SHAP")