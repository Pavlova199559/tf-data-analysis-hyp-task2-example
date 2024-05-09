import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from google.colab import files
chat_id = 789271490  # Ваш chat ID, не меняйте название переменной

# Загрузка данных
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
data = pd.read_csv(file_name)

# Разделяем контрольную и тестовую группы
control = data[data['ID'] % 2 == 0]['Флаг утилизации счёта'].values
test = data[data['ID'] % 2 != 0]['Флаг утилизации счёта'].values

# Проверка, что массивы не пустые
assert len(control) > 0, "Контрольная группа пуста"
assert len(test) > 0, "Тестовая группа пуста"

# Проверка, что массивы содержат только числовые данные
assert np.issubdtype(control.dtype, np.number), "Контрольная группа содержит нечисловые данные"
assert np.issubdtype(test.dtype, np.number), "Тестовая группа содержит нечисловые данные"

# Вывод данных для проверки
print("Контрольная группа:", control[:10])  # Печать первых 10 значений
print("Тестовая группа:", test[:10])

# Функция, которая использует тест Манна-Уитни для проверки различий
def solution(x: np.array, y: np.array) -> bool:
    alpha = 0.02
    stat, p_value = mannwhitneyu(x, y, alternative='two-sided')
    return p_value <= alpha

# Применение функции к контрольной и тестовой группам
result = solution(control, test)

print("Статистически значимая разница:", result)  # True, если есть разница, False, если нет
