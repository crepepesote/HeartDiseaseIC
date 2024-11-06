import pandas as pd

# Cargar el dataset
dataset = pd.read_csv("C:/Users/Luis Alfonso/Pictures/Project/Output/balanced_heart_disease_dataset.csv")
  # Cambia la ruta a donde tengas el dataset

# Revisar el desbalance en la columna de 'target'
print(dataset['target'].value_counts())
