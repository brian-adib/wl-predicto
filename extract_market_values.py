import pandas as pd
import os

# Ruta al archivo test.csv (cámbiala si está en otra carpeta)
test_csv_path = "data/raw/test.csv"   # porque está en data/raw/
# Cargar el CSV
df = pd.read_csv(test_csv_path)

# Ver nombres de columnas para identificar la de valor de mercado
print("Columnas disponibles:", df.columns.tolist())

# Buscar la columna que contiene el valor de mercado
# Por la captura, puede llamarse 'market_value' o 'value'
value_col = None
for col in df.columns:
    if 'market' in col.lower() or 'value' in col.lower():
        value_col = col
        break

if value_col is None:
    raise ValueError("No se encontró columna de valor de mercado. Revisa los nombres.")

print(f"Usando columna: {value_col}")

# Seleccionar solo las columnas 'team' y la de valor
team_values = df[['team', value_col]].copy()
team_values.columns = ['team', 'market_value_eur']

# Guardar en data/processed/
output_path = "data/processed/team_market_values.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
team_values.to_csv(output_path, index=False)

print(f"✅ Archivo guardado en {output_path}")
print(team_values.head())