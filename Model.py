import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, confusion_matrix
import tkinter as tk
from tkinter import scrolledtext

# Cargar el dataset
dataset = pd.read_csv("C:/Users/Luis Alfonso/Pictures/Project/Dataset/heart.csv")  # Cambia la ruta a tu archivo

# Separar las características y la variable objetivo
X = dataset.drop('target', axis=1)
y = dataset['target']

# Dividir en entrenamiento y prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo y entrenarlo
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluar el modelo en el conjunto de prueba
y_test_pred = model.predict(X_test)

# Calcular precisión
precision = precision_score(y_test, y_test_pred)

# Generar matriz de confusión
test_cm = confusion_matrix(y_test, y_test_pred)

# Crear la ventana principal
root = tk.Tk()
root.title("Resultados de Clasificación")

# Crear un área de texto con scroll para mostrar el reporte
output_text = scrolledtext.ScrolledText(root, width=80, height=10, wrap=tk.WORD)
output_text.grid(row=0, column=0, padx=10, pady=10)

# Agregar los resultados al área de texto
output_text.insert(tk.END, f"Precisión en el conjunto de prueba: {precision:.4f}\n")
output_text.insert(tk.END, "\nMatriz de Confusión en Prueba:\n")
output_text.insert(tk.END, str(test_cm))

# Hacer que la ventana sea interactiva
root.mainloop()
