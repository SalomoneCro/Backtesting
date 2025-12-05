import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# --- Configuración y Carga de Datos ---
def load_data(filepath: str) -> pd.DataFrame:
    """Carga el dataset y asegura el formato correcto de las fechas."""
    try:
        df = pd.read_csv(filepath, parse_dates=['Date'])
        print(f"Dataset cargado desde: {filepath}")
        print(f"Total de filas disponibles para el entrenamiento: {len(df)}")
        return df
    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo en la ruta: {filepath}")
        print("Asegúrate de haber ejecutado primero DataFetcher para generar el CSV.")
        return pd.DataFrame()

# --- Preprocesamiento de Datos ---
def preprocess_data(df: pd.DataFrame):
    """
    Prepara los datos para el entrenamiento:
    1. Define features (X) y target (y).
    2. Escala las features numéricas.
    """
    
    # 1. Definir la variable objetivo (y) y las características (X)
    
    y = df['target']
    
    # Excluimos columnas no predictivas y la target del conjunto de features (X)
    X = df.drop(columns=['Date', 'target'])
    
    # Identificar las columnas numéricas para escalado (todas las que quedan)
    numeric_cols = X.columns.tolist()

    # 2. Escalado de datos (Normalización)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[numeric_cols])
    
    X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols, index=X.index)
    
    print(f"\nFeatures (X) lista. Columnas usadas: {len(numeric_cols)}")
    print("Datos numéricos escalados usando StandardScaler.")
    
    return X_scaled_df, y, scaler

# --- Entrenamiento del Modelo ---
def train_model(X_train, y_train):
    """
    Entrena un modelo Gradient Boosting Classifier para CLASIFICACIÓN BINARIA (0 o 1).
    """
    print("\nIniciando entrenamiento del Gradient Boosting Classifier (Clasificación Binaria)...")
    
    # El mismo modelo funciona para binario y multiclase, no requiere cambios de parámetro.
    model = GradientBoostingClassifier(
        n_estimators=1500,      
        learning_rate=0.1,     
        max_depth=8,           
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print("✓ Entrenamiento completado.")
    
    return model

# --- Evaluación del Modelo ---
def evaluate_model(model, X_test, y_test):
    """
    Realiza predicciones y evalúa el modelo, enfocado en el problema binario.
    """
    print("\n--- Evaluación del Modelo ---")
    
    # Predicción en el conjunto de prueba
    y_pred = model.predict(X_test)
    
    # Métricas de rendimiento
    accuracy = accuracy_score(y_test, y_pred)
    
    # Para el reporte de clasificación, es útil especificar las etiquetas 
    # si queremos que el informe sea más descriptivo.
    target_names = ['0 (Baja/Igual)', '1 (Sube)']
    
    conf_mat = confusion_matrix(y_test, y_pred)
    
    # Incluimos 'labels' y 'target_names' para claridad en el informe
    report = classification_report(
        y_test, 
        y_pred, 
        labels=[0, 1], # Aseguramos que solo se evalúen estas dos clases
        target_names=target_names, 
        zero_division=0
    )
    
    print(f"Accuracy (Precisión General): {accuracy:.4f}")
    print("\nMatriz de Confusión (Confusion Matrix):")
    print("Fila = Real (True), Columna = Predicción (Predicted)")
    print("Clases: 0 (Baja/Igual), 1 (Sube)")
    print(conf_mat)
    print("\nReporte de Clasificación (Classification Report):")
    print(report)

# --- Función Principal ---
def main():
    # RUTA: Ajusta esto a la ruta de tu archivo CSV generado por DataFetcher
    # Ejemplo:
    FILE_PATH = os.path.join("ExploratoryAnalysis", "Datasets", "AAPL_2021-12-31_2025-01-01.csv") 

    # 1. Cargar datos
    data = load_data(FILE_PATH)
    if data.empty:
        return

    # 2. Preprocesar datos
    X, y, scaler = preprocess_data(data)
    
    # Usamos Time Series Split (sin shuffle)
    split_index = int(len(X) - 20)
    
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    
    print(f"\nSeparación de conjuntos:")
    print(f"  Entrenamiento: {len(X_train)} filas")
    print(f"  Prueba: {len(X_test)} filas")
    
    # 4. Entrenar modelo
    model = train_model(X_train, y_train)

    # 5. Evaluar modelo
    evaluate_model(model, X_test, y_test)
    
    print("\nEl modelo ahora está entrenado para predecir si el precio SUBIRÁ (1) o NO (0).")


if __name__ == "__main__":
    main()