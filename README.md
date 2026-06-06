# 📈 Retail Turnover (RTO) Forecasting Pipeline

Данный репозиторий содержит решение для задачи прогнозирования розничного товарооборота (РТО) на месяц вперед. Модель построена на основе ансамбля (блендинга) трех алгоритмов с предварительной кластеризацией объектов.

## 🚀 Особенности решения

*   **Кластеризация магазинов**: Использование иерархической кластеризации (метод Уорда) для группировки магазинов по профилю продаж, уровню CV (коэффициент вариации) и трендам.
*   **Глубокий Feature Engineering**:
    *   Лаги (1, 2, 3, 6, 12, 13, 24 месяца).
    *   Скользящие средние и экспоненциальное сглаживание (EMA/EWM).
    *   YoY (Year-over-Year) показатели и темпы роста.
    *   Тригонометрические признаки для сезонности (sin/cos месяцев).
    *   Агрегаты по кластерам и регионам.
*   **Ансамбль моделей**:
    *   **LightGBM**: Градиентный бустинг, оптимизированный через Optuna.
    *   **MLP (Multi-layer Perceptron)**: Нейронная сеть для улавливания нелинейных зависимостей.
    *   **Ridge Regression**: Линейная модель для повышения стабильности прогноза.
*   **Автоматический подбор гиперпараметров**: Использование библиотеки Optuna для поиска оптимальных параметров бустинга и регрессии.

## 🛠 Технологический стек

*   **Python 3.x**
*   **Data Science**: Pandas, NumPy, Scipy, Scikit-learn
*   **ML**: LightGBM
*   **Optimization**: Optuna

## 📋 Структура пайплайна

1.  **Загрузка данных**: Преобразование даты и предобработка исходного файла.
2.  **Clustering**: Создание профилей магазинов и разделение их на кластеры для учета специфики (крупные/мелкие, стабильные/волатильные).
3.  **Feature Building**: Генерация более 50 признаков для временных рядов.
4.  **HPO (Hyperparameter Optimization)**: Запуск Optuna для LightGBM и Ridge для минимизации MAPE.
5.  **Training & Prediction**: Обучение финальных моделей на всей доступной выборке.
6.  **Blending**: Взвешенное объединение предсказаний:
    *   `0.44 * LightGBM + 0.51 * MLP + 0.05 * Ridge`
7.  **Post-processing**: Применение специфических коэффициентов коррекции для разных сегментов выручки.

## 💻 Запуск

1. Установите зависимости:
   ```bash
   pip install pandas numpy lightgbm scikit-learn scipy optuna
   ```

2. Разместите файл с данными `train_2.csv` в корневой папке.

3. Запустите скрипт:
   ```bash
   python final.ipynb
   ```

Результат будет сохранен в файл `submission.csv`.

## 📊 Формула оценки

Качество модели оценивается по метрике MAPE (Mean Absolute Percentage Error) с последующей конвертацией в баллы конкурса:
$$Score = 100 \times \left(\frac{100 - \min(MAPE, 100)}{100}\right)^2$$

### Итоговые показатели:
- **Результат на Leaderboard:** `90.45`
- **Стабильность модели:** Высокая (достигнута за счет тройного блендинга моделей и предварительной кластеризации магазинов).

## ⚙️ Константы и настройки

*   `TARGET_DATE`: `2025-03-01` (Прогнозный период)
*   `VALID_DATE`: `2025-02-01` (Валидационный отсечка)
*   `N_TRIALS`: 100 (Количество итераций Optuna)

## 👥 Команда проекта

| Name | GitHub |
| :--- | :--- |
| **Владислав** | [@Vladislav585](https://github.com/Vladislav585) |
| **Лев** | [@KremlevLev](https://github.com/KremlevLev) |

---


# 📈 Retail Turnover (RTO) Forecasting Pipeline

This repository contains a high-performance solution for predicting retail turnover (RTO) for the next month. The system utilizes a sophisticated ensemble of Gradient Boosting, Neural Networks, and Linear models, optimized through automated hyperparameter tuning.

## 🚀 Key Features

*   **Store Clustering**: Multi-level hierarchical clustering (Ward's linkage) based on sales profiles, coefficient of variation (CV), and trend analysis to group similar stores.
*   **Advanced Feature Engineering**: 
    *   **Time-series lags**: 1, 2, 3, 6, 12, 13, 24-month shifts.
    *   **Statistical Windows**: Rolling means, standard deviations, and Exponential Weighted Moving Averages (EWM).
    *   **Seasonality**: Trigonometric encoding (Sine/Cosine) of months and specific holiday flags (e.g., March peaks).
    *   **External Aggregates**: Regional and cluster-based target encoding.
*   **Automated HPO**: Integration with **Optuna** for Bayesian optimization of LightGBM and Ridge regression parameters.
*   **Model Blending**: A weighted ensemble designed for stability:
    *   `0.44 * LightGBM` + `0.51 * MLP` + `0.05 * Ridge`.

## 🛠 Tech Stack

*   **Language**: Python 3.9+
*   **ML Frameworks**: LightGBM, Scikit-learn (MLPRegressor, Ridge)
*   **Optimization**: Optuna
*   **Data Science**: Pandas, NumPy, Scipy (Hierarchical Clustering)

## 📋 Pipeline Architecture

1.  **Preprocessing**: Date conversion, RTO renaming, and basic cleaning.
2.  **Clustering**: Stores are segmented into 4 primary levels by volume, then sub-clustered using Ward's method to capture behavior patterns.
3.  **Feature Construction**: Generation of over 60 temporal and categorical features.
4.  **LGBM Optimization**: 100 trials of Optuna searching for optimal tree depth, learning rate, and regularization.
5.  **Neural Network (MLP)**: A 3-layer MLP focused on capturing non-linear relationships using scaled historical data.
6.  **Ridge Regression**: A stabilized linear component to reduce variance in the final ensemble.
7.  **Blending & Correction**: Final predictions are blended and passed through a custom post-processing function (`apply_coef`) to adjust for specific revenue brackets.

## 📊 Evaluation Metric

The model is optimized for **MAPE** (Mean Absolute Percentage Error). The competition score is calculated as:

$$Score = 100 \times \left(\frac{100 - \min(MAPE, 100)}{100}\right)^2$$

### Final Performance:
- **Leaderboard Score:** `90.45`
- **Model Stability:** High (achieved through triple blending and clustering).

## 💻 Installation & Usage

1. **Clone the repo**:
   ```bash
   git clone https://github.com/Vladislav585/X5-Pyaterochka-RTO-forecast-second-stage.git
   cd X5-Pyaterochka-RTO-forecast-second-stage
   ```

2. **Install dependencies**:
   ```bash
   pip install pandas numpy lightgbm scikit-learn scipy optuna
   ```

3. **Run the pipeline**:
   Ensure `train_2.csv` is in the root directory, then run:
   ```bash
   python final.ipynb
   ```
   The script will generate `submission.csv` with the final predictions.

## 👥 Team

Developed by:

| Name | GitHub |
| :--- | :--- |
| **Vladislav** | [@Vladislav585](https://github.com/Vladislav585) |
| **Lev** | [@KremlevLev](https://github.com/KremlevLev) |

---
*This project was developed for retail data analytics and time-series forecasting challenges.*
