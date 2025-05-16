import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import folium
from folium.plugins import HeatMap
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Configuramos el estilo de las visualizaciones
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Aumentamos el tamaño de las figuras para mejor visualización
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("# 1. Carga y Limpieza de Datos")
print("=" * 50)

# Cargamos el dataset 
try:
    # Intentar cargar desde un archivo CSV
    df = pd.read_csv('Airbnb_Open_Data.csv')
    print(f"Dataset cargado con éxito: {df.shape[0]} filas y {df.shape[1]} columnas")
except:
    print("No se encontró el archivo. Por favor intenta nuevamente con otro archivo")

    


print("\n# 1.1 Detección y tratamiento de datos faltantes")
print("=" * 50)

# Verificamos los valores faltantes por columna
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({
    'Valores faltantes': missing_values,
    'Porcentaje (%)': missing_percentage
})

print("Valores faltantes por columna:")
print(missing_df[missing_df['Valores faltantes'] > 0].sort_values('Valores faltantes', ascending=False))

# Tratamiento de valores faltantes
print("\nTratamiento de valores faltantes:")

# Para datos numéricos: se imputar con la mediana
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_value = df[col].median()
        df[col].fillna(median_value, inplace=True)
        print(f"{col}': {missing_values[col]} valores faltantes imputados con la mediana ({median_value:.2f})")

# Para datos categóricos: imputar con el valor más frecuente
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_value = df[col].mode()[0]
        df[col].fillna(mode_value, inplace=True)
        print(f"- Columna '{col}': {missing_values[col]} valores faltantes imputados con el modo ({mode_value})")

# Verificamos que no queden valores faltantes
missing_after = df.isnull().sum().sum()
print(f"\nValores faltantes después del tratamiento: {missing_after}")

print("\n# 1.2 Transformación de columnas útiles")
print("=" * 50)


# Crear categorías de precio para análisis

#Limpiamos los datos en caso de que tengan datos no validos
df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
df['price_category'] = pd.qcut(df['price'],q=4,labels=['Economico','Medio','Alto','Premium'])

print("\n# 1.3 Identificación de outliers")
print("=" * 50)

# Identificar outliers en el precio
Q1 = df['price'].quantile(0.25) 
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]
print(f"Outliers en precio (Método IQR): {len(outliers)} registros")
print(f"- Límite inferior: {lower_bound:.2f}")
print(f"- Límite superior: {upper_bound:.2f}")

# Visualizar outliers en precio con boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(x=df['price'])
plt.title('Identificación de Outliers en Precio')
plt.tight_layout()
plt.show()


print("\n# 1.4 Preparación de variables para el análisis")
print("=" * 50)

# Seleccionar las características relevantes para los análisis posteriores

df['room_type_graf']=df['room type'].map({'Entire home/apt':0, 'Private room':1})

df['features'] = df['price'].astype(str) + ' | ' + df['room type']




if 'property_type' in df.columns:
    df['features'].append('property_type')
if 'accommodates' in df.columns:
    df['features'].append('accommodates')
if 'is_instant_bookable' in df.columns:
    df['features'].append('is_instant_bookable')

# Crear un dataframe con las características seleccionadas

df_analysis = df[['features']].copy()
print(f"Dataset preparado para análisis con {df_analysis.shape[1]} variables seleccionadas:")
print(df_analysis.columns.tolist())

# Verificar que no haya valores faltantes en el conjunto final
missing_final = df_analysis.isnull().sum().sum()
print(f"\nValores faltantes en el conjunto final: {missing_final}")

# Estadísticas descriptivas del conjunto final
print("\nEstadísticas descriptivas del conjunto final:")
print(df_analysis.describe())

print("\n# 2. Análisis Exploratorio con Visualizaciones")
print("=" * 50)

print("\n# 2.1 Comparación de precios por tipo de alojamiento")
print("=" * 50)

# Gráfico de violín para la distribución de precios por tipo de alojamiento
plt.figure(figsize=(14, 8))
sns.violinplot(x='room_type_graf', y='price', data=df, palette='viridis')
plt.title('Distribución de Precios por Tipo de Alojamiento', fontsize=16)
plt.xlabel('Tipo de Alojamiento', fontsize=14)
plt.ylabel('Precio (€)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Estadísticas descriptivas de precio por tipo de alojamiento
price_by_room_type = df.groupby('room_type_graf')['price'].agg(['count', 'mean', 'median', 'min', 'max'])
print("Estadísticas de precio por tipo de alojamiento:")
print(price_by_room_type)

# Si existe la columna property_type
if 'property_type' in df.columns:
    # Calcular el precio promedio por tipo de propiedad
    price_by_property = df.groupby('property_type')['price'].mean().sort_values(ascending=False)
    
    # Mostrar un gráfico de barras con los 10 tipos de propiedad más caros en promedio
    plt.figure(figsize=(14, 8))
    price_by_property.head(10).plot(kind='bar', color=sns.color_palette("viridis", 10))
    plt.title('Precio Promedio por Tipo de Propiedad (Top 10)', fontsize=16)
    plt.xlabel('Tipo de Propiedad', fontsize=14)
    plt.ylabel('Precio Promedio (€)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    print("\nPrecio promedio por tipo de propiedad (Top 10):")
    print(price_by_property.head(10))

print("\n# 2.2 Mapa de calor o gráfico de correlación de variables")
print("=" * 50)

# Seleccionar solo variables numéricas para la matriz de correlación
numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')


# Calcular la matriz de correlación
corr_matrix = numeric_df.corr()

# Visualizar la matriz de correlación
plt.figure(figsize=(14, 10))
mask = np.triu(corr_matrix)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='viridis', mask=mask,
            linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('Matriz de Correlación de Variables Numéricas', fontsize=16)
plt.xticks(fontsize=12, rotation=45, ha='right')
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

print("Observaciones sobre las correlaciones:")
# Identificar las correlaciones más fuertes (positivas y negativas) con el precio
price_correlations = corr_matrix['price'].sort_values(ascending=False)
print("Correlaciones con el precio:")
print(price_correlations)

print("\n# 2.3 Segmentación de alojamientos por rangos de precio y ubicación")
print("=" * 50)

# Segmentación por ubicación (neighbourhood_group)
price_by_neighborhood = df.groupby('neighbourhood')['price'].agg(['count', 'mean', 'median'])
price_by_neighborhood = price_by_neighborhood.sort_values('mean', ascending=False)

# Visualizar precio promedio por vecindario
plt.figure(figsize=(14, 8))
sns.barplot(x=price_by_neighborhood.index, y=price_by_neighborhood['mean'], 
            palette=sns.color_palette("viridis", len(price_by_neighborhood)))
plt.title('Precio Promedio por Vecindario', fontsize=16)
plt.xlabel('Vecindario', fontsize=10)
plt.ylabel('Precio Promedio (€)', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

print("Precio promedio por vecindario:")
print(price_by_neighborhood)

# Crear mapa de calor de precios por ubicación
print("\nGenerando mapa de calor de precios por ubicación...")

# Crear un dataframe para el mapa de calor con latitud, longitud y precio
heat_df = df[['lat', 'long', 'price']].copy()

# Crear un mapa centrado en las coordenadas promedio
center_lat = heat_df['lat'].mean()
center_lon = heat_df['long'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Añadir capa de calor basada en los precios
heat_data = [[row['lat'], row['long'], row['price']] for idx, row in heat_df.iterrows()]
# Asegúrate de que las columnas 'latitude' y 'longitude' sean numéricas
df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
df['long'] = pd.to_numeric(df['long'], errors='coerce')

gradient = {'0.2': 'blue', '0.4': 'green', '0.6': 'yellow', '0.8': 'orange', '1.0': 'red'}
HeatMap(heat_data, radius=15, gradient=gradient).add_to(m)

# Guardar el mapa como HTML
m.save('heatmap_precios_airbnb.html')
print("Mapa de calor guardado como 'heatmap_precios_airbnb.html'")

# Segmentación por rangos de precio y tipo de habitación
plt.figure(figsize=(14, 8))
sns.countplot(x='price_category', hue='room_type_graf', data=df, palette='viridis')
plt.title('Distribución de Tipos de Alojamiento por Categoría de Precio', fontsize=16)
plt.xlabel('Categoría de Precio', fontsize=14)
plt.ylabel('Número de Alojamientos', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Tipo de Alojamiento', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

print("\n# 2.4 Otras visualizaciones relevantes")
print("=" * 50)




# 1. Relación entre el precio y el número de reseñas
plt.figure(figsize=(12, 8))
sns.scatterplot(x='number of reviews', y='price', hue='room_type_graf', data=df, alpha=0.6, palette='viridis')
plt.title('Relación entre Precio y Número de Reseñas', fontsize=16)
plt.xlabel('Número de Reseñas', fontsize=14)
plt.ylabel('Precio (€)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Tipo de Alojamiento', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2. Distribución de la disponibilidad durante el año
plt.figure(figsize=(12, 8))
sns.histplot(df['availability 365'], bins=30, kde=True, color='darkblue')
plt.title('Distribución de la Disponibilidad Anual', fontsize=16)
plt.xlabel('Días Disponibles en el Año', fontsize=14)
plt.ylabel('Número de Alojamientos', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3. Análisis de clustering para identificar patrones
# Seleccionar características para clustering
if 'accommodates' in df.columns and 'number_of_bedrooms' in df.columns:
    cluster_features = df[['price', 'accommodates', 'number_of_bedrooms']].copy()
    
    # Escalar las características
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_features)
    
    # Aplicar K-means
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['cluster'] = kmeans.fit_predict(scaled_features)
    
    # Visualizar los clusters
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='accommodates', y='price', hue='cluster', data=df, palette='viridis', s=100, alpha=0.7)
    plt.title('Clustering de Alojamientos por Precio y Capacidad', fontsize=16)
    plt.xlabel('Capacidad (Personas)', fontsize=14)
    plt.ylabel('Precio (€)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Cluster', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Analizar características de cada cluster
    cluster_analysis = df.groupby('cluster').agg({
        'price': ['mean', 'median'],
        'accommodates': 'mean',
        'number_of_bedrooms': 'mean',
        'number_of_reviews': 'mean'
    })
    
    print("Análisis de clusters:")
    print(cluster_analysis)

print("\n# 3. Modelo de Predicción de Precios")
print("=" * 50)

print("\n# 3.1 Explicación de variables utilizadas")
print("=" * 50)

# Seleccionar características para el modelo
# Para este ejemplo, usaremos características que suelen tener alta correlación con el precio
model_features = [
    'room_type_graf',              # Tipo de habitación (categórica)
    'neighbourhood group',    # Zona/vecindario (categórica)
    'availability 365',       # Disponibilidad anual (numérica)
    'number of reviews'       # Número de reseñas (numérica)
]

# Añadir otras características si están disponibles
if 'accommodates' in df.columns:
    model_features.append('accommodates')  # Capacidad (numérica)
if 'number_of_bedrooms' in df.columns:
    model_features.append('number_of_bedrooms')  # Número de dormitorios (numérica)
if 'number_of_bathrooms' in df.columns:
    model_features.append('number_of_bathrooms')  # Número de baños (numérica)
if 'is_instant_bookable' in df.columns:
    model_features.append('is_instant_bookable')  # Reserva instantánea (binaria)

print("Variables seleccionadas para el modelo:")
for feature in model_features:
    if feature in df.select_dtypes(include=[np.number]).columns:
        print(f"- {feature} (numérica)")
    else:
        print(f"- {feature} (categórica)")

print("\nJustificación de las variables seleccionadas:")
print("- room_type: El tipo de alojamiento es un fuerte determinante del precio")
print("- neighbourhood_group: La ubicación afecta significativamente el precio")
print("- availability_365: La disponibilidad puede reflejar la demanda")
print("- number_of_reviews: Puede ser un indicador de popularidad")


if 'accommodates' in model_features:
    print("- accommodates: La capacidad está directamente relacionada con el tamaño y por tanto con el precio")
if 'number_of_bedrooms' in model_features:
    print("- number_of_bedrooms: Más dormitorios generalmente implican mayor precio")
if 'number_of_bathrooms' in model_features:
    print("- number_of_bathrooms: Más baños suelen significar mayor comodidad y mayor precio")
if 'is_instant_bookable' in model_features:
    print("- is_instant_bookable: Puede influir en la conveniencia y por tanto en el precio")

print("\n# 3.2 Divisiones de entrenamiento/prueba")
print("=" * 50)

# Preparar X (características) e y (variable objetivo)
X = df[model_features].copy()
y = df['price']

# Identificar columnas numéricas y categóricas
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"Características numéricas: {numeric_features}")
print(f"Características categóricas: {categorical_features}")

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"Tamaño del conjunto de prueba: {X_test.shape[0]} muestras")

# Crear preprocesadores para características numéricas y categóricas
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Crear un preprocesador columnar que aplique las transformaciones adecuadas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

print("\n# 3.3 Entrenamiento del modelo")
print("=" * 50)

# Crear y entrenar un modelo de Random Forest
model_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

print("Entrenando modelo Random Forest...")
model_rf.fit(X_train, y_train)
print("Modelo entrenado con éxito.")

# También entrenar un modelo de regresión lineal para comparar
model_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

print("\nEntrenando modelo de Regresión Lineal para comparación...")
model_lr.fit(X_train, y_train)
print("Modelo de Regresión Lineal entrenado con éxito.")

print("\n# 3.4 Evaluación del modelo")
print("=" * 50)

# Función para evaluar el modelo
def evaluate_model(model, X_test, y_test, model_name):
    # Hacer predicciones
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Resultados para {model_name}:")
    print(f"- RMSE (Error Cuadrático Medio): {rmse:.2f}")
    print(f"- MAE (Error Absoluto Medio): {mae:.2f}")
    print(f"- R² (Coeficiente de Determinación): {r2:.4f}")
    
    # Visualizar predicciones vs valores reales
    plt.figure(figsize=(12, 8))
    plt.scatter(y_test, y_pred, alpha=0.5, color='darkblue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f'Valores Reales vs Predicciones ({model_name})', fontsize=16)
    plt.xlabel('Precio Real (€)', fontsize=14)
    plt.ylabel('Precio Predicho (€)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return rmse, mae, r2, y_pred

# Evaluar el modelo Random Forest
print("Evaluación del modelo Random Forest:")
rmse_rf, mae_rf, r2_rf, y_pred_rf = evaluate_model(model_rf, X_test, y_test, "Random Forest")

# Evaluar el modelo de Regresión Lineal
print("\nEvaluación del modelo de Regresión Lineal:")
rmse_lr, mae_lr, r2_lr, y_pred_lr = evaluate_model(model_lr, X_test, y_test, "Regresión Lineal")

# Comparar los modelos
print("\nComparación de modelos:")
models_comparison = pd.DataFrame({
    'Modelo': ['Random Forest', 'Regresión Lineal'],
    'RMSE': [rmse_rf, rmse_lr],
    'MAE': [mae_rf, mae_lr],
    'R²': [r2_rf, r2_lr]
})
print(models_comparison)

print("\n# 3.5 Análisis de importancia de variables")
print("=" * 50)
