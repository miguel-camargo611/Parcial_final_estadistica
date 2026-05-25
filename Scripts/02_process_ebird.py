import pandas as pd
import numpy as np
import time

def main():
    start_time = time.time()
    ebird_file = r'Data\Raw\ebird\ebd_CO-DC_smp_relMar-2026\ebd_CO-DC_smp_relMar-2026.txt'
    
    print("Leyendo archivo eBird...")
    cols_to_use = [
        'SAMPLING EVENT IDENTIFIER', 'OBSERVATION DATE', 'TIME OBSERVATIONS STARTED',
        'LATITUDE', 'LONGITUDE', 'DURATION MINUTES', 'EFFORT DISTANCE KM', 
        'ALL SPECIES REPORTED', 'COMMON NAME', 'OBSERVATION COUNT'
    ]
    
    df = pd.read_csv(ebird_file, sep='\t', on_bad_lines='skip', low_memory=False, usecols=cols_to_use)
    
    print(f"Total registros iniciales: {len(df)}")
    
    # 1. Filtro Temporal: periodo de análisis 2020-2025
    df['OBSERVATION DATE'] = pd.to_datetime(df['OBSERVATION DATE'], errors='coerce')
    df = df[(df['OBSERVATION DATE'] >= '2020-01-01') & (df['OBSERVATION DATE'] < '2026-01-01')]
    
    # 2. Filtro Listas Completas
    df = df[df['ALL SPECIES REPORTED'] == 1]
    
    # 3. Limpiar Conteos (Eliminar 'X')
    # Identificar listas que tienen al menos una 'X'
    def is_numeric(val):
        try:
            float(val)
            return True
        except:
            return False
            
    df['is_numeric'] = df['OBSERVATION COUNT'].apply(is_numeric)
    listas_con_x = df[~df['is_numeric']]['SAMPLING EVENT IDENTIFIER'].unique()
    
    # Quedarnos solo con listas que son 100% numéricas
    df_clean = df[~df['SAMPLING EVENT IDENTIFIER'].isin(listas_con_x)].copy()
    df_clean['OBSERVATION COUNT'] = df_clean['OBSERVATION COUNT'].astype(float)
    
    print(f"Listas viables tras filtros: {df_clean['SAMPLING EVENT IDENTIFIER'].nunique()}")
    
    # 4. Calcular el Índice de Shannon por Lista
    print("Calculando Índice de Shannon...")
    
    def shannon_index(counts):
        p = counts / counts.sum()
        p = p[p > 0] # evitar log(0)
        return -np.sum(p * np.log(p))
        
    shannon_series = df_clean.groupby('SAMPLING EVENT IDENTIFIER')['OBSERVATION COUNT'].apply(shannon_index)
    shannon_series.name = 'Shannon_Index'
    
    riqueza_series = df_clean.groupby('SAMPLING EVENT IDENTIFIER')['COMMON NAME'].nunique()
    riqueza_series.name = 'Riqueza_Especies'
    
    # 5. Extraer metadata por lista (tomamos el primer registro de cada lista, ya que lat, lon, date son iguales)
    metadata = df_clean.drop_duplicates(subset=['SAMPLING EVENT IDENTIFIER']).set_index('SAMPLING EVENT IDENTIFIER')
    metadata = metadata[['OBSERVATION DATE', 'TIME OBSERVATIONS STARTED', 'LATITUDE', 'LONGITUDE', 'DURATION MINUTES', 'EFFORT DISTANCE KM']]
    
    # 6. Unir todo
    final_ebird = metadata.join(shannon_series).join(riqueza_series).reset_index()
    
    # Parsear hora. eBird suele traer HH:MM:SS, pero algunos registros pueden
    # venir en otros formatos; convertimos de forma tolerante sin agregar sufijos.
    time_as_text = final_ebird['TIME OBSERVATIONS STARTED'].astype('string').str.strip()
    final_ebird['TIME OBSERVATIONS STARTED'] = pd.to_timedelta(time_as_text, errors='coerce')
    final_ebird['Fecha_Hora'] = final_ebird['OBSERVATION DATE'] + final_ebird['TIME OBSERVATIONS STARTED']
    
    # Truncar a la hora para cruzar con RMCAB
    final_ebird['Fecha_Hora_Truncada'] = final_ebird['Fecha_Hora'].dt.floor('h')
    
    output_path = r"Data\Processed\ebird_shannon.parquet"
    final_ebird.to_parquet(output_path, engine='pyarrow', index=False)
    
    print(f"Guardado exitosamente en: {output_path}")
    print(f"Tiempo total: {(time.time() - start_time)/60:.2f} minutos")

if __name__ == "__main__":
    main()
