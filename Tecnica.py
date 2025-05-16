import pandas as pd

# Se guarda el archivos csv dentro de una variable para poder manipular los datos 
archivo = "Airbnb_Open_Data.csv"

# Cargar el dataset
df = pd.read_csv(archivo, low_memory=False)

df_floats = df[df['price'].apply(lambda x: isinstance(x, float))]

# Mostrar el resultado
print(df_floats)