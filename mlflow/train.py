import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor  # Изменили импорт на регрессор, так как у нас задача регрессии
from sklearn.metrics import mean_squared_error  # Используем метрику среднеквадратической ошибки для регрессии
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, Trials

mlflow.set_tracking_uri("http://localhost:5000")

# Загрузка данных о ценах на недвижимость в Калифорнии
df = pd.read_csv("California_Houses.csv")  # Предполагается, что файл находится в той же папке, что и скрипт

# Обработка пропущенных значений
df = df.dropna()

# Разделение данных на обучающий и тестовый наборы
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Определение признаков и целевой переменной
features = ["Median_Income", "Median_Age", "Tot_Rooms", "Tot_Bedrooms", "Population", "Households", "Latitude", "Longitude", "Distance_to_coast", "Distance_to_LA", "Distance_to_SanDiego", "Distance_to_SanJose", "Distance_to_SanFrancisco"]
target = "Median_House_Value"

# Предобработка данных
train_X = train[features]
train_y = train[target]
test_X = test[features]
test_y = test[target]

def objective(params):
    clf = RandomForestRegressor(**params)  # Используем регрессор
    clf.fit(train_X, train_y)
    y_pred = clf.predict(test_X)
    score = mean_squared_error(test_y, y_pred)  # Метрика среднеквадратической ошибки для регрессии
    return score

space = {
    'n_estimators': hp.choice('n_estimators', range(10, 200)),
    'max_depth': hp.choice('max_depth', range(1, 20)),
    'max_features': hp.choice('max_features', range(1, 9))
}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

model = RandomForestRegressor(n_estimators=best['n_estimators'] + 10,
                              max_depth=best['max_depth'] + 1,
                              max_features=best['max_features'] + 1)
model.fit(train_X, train_y)
y_pred = model.predict(test_X)

# Оценка модели на тестовом наборе
mse = mean_squared_error(test_y, y_pred)

# Логирование параметров модели и метрик
with mlflow.start_run(experiment_id=1):
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("data", "California_Houses")
    mlflow.log_metric("mean_squared_error", mse)
    mlflow.sklearn.log_model(model, "model")