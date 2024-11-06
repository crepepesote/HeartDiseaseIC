import pandas as pd
from imblearn.over_sampling import SMOTE

# Cargar el dataset original
dataset = pd.read_csv("C:/Users/Luis Alfonso/Pictures/Project/Dataset/heart.csv")  # Cambia la ruta a donde tengas el archivo

# Separar las características y la variable objetivo
X = dataset.drop('target', axis=1)
y = dataset['target']

# Crear una instancia de SMOTE
# Calculamos el número de datos que necesitamos generar para cada clase
smote = SMOTE(sampling_strategy={0: len(X[y == 0]) + 500, 1: len(X[y == 1]) + 500}, random_state=42)

# Aplicar SMOTE para balancear las clases
X_balanced, y_balanced = smote.fit_resample(X, y)

# Verificar el nuevo balance
print(y_balanced.value_counts())

# Combinar características y target nuevamente en un DataFrame
balanced_dataset = pd.concat([X_balanced, y_balanced], axis=1)

# Guardar el nuevo dataset balanceado (opcional)
balanced_dataset.to_csv("C:/Users/Luis Alfonso/Pictures/Project/Output/balanced_heart_disease_dataset.csv", index=False)

