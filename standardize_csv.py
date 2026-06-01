import pandas as pd
import os

def standardize_results(input_path="data/raw/results.csv", output_path="data/processed/results_clean.csv"):
    print("Estandarizando results.csv...")
    df = pd.read_csv(input_path)
    # Convertir fecha a datetime con manejo de formatos
    df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False, errors='coerce')
    # Eliminar filas con fecha nula
    before = len(df)
    df = df.dropna(subset=['date'])
    after = len(df)
    print(f"  Fechas inválidas eliminadas: {before - after}")
    # Ordenar por fecha
    df = df.sort_values('date')
    # Guardar
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Guardado en {output_path}")
    return df

def standardize_eloratings(input_path="data/raw/eloratings.csv", output_path="data/processed/eloratings_clean.csv"):
    print("Estandarizando eloratings.csv...")
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False, errors='coerce')
    before = len(df)
    df = df.dropna(subset=['date'])
    after = len(df)
    print(f"  Fechas inválidas eliminadas: {before - after}")
    df = df.sort_values('date')
    # Asegurar columna 'rating' existe y renombrar a 'elo' (opcional)
    if 'rating' in df.columns:
        df = df.rename(columns={'rating': 'elo'})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Guardado en {output_path}")
    return df

if __name__ == "__main__":
    standardize_results()
    standardize_eloratings()
    print("✅ Estandarización completada. Ahora puedes usar los archivos en data/processed/")