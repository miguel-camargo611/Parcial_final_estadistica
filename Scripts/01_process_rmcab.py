import os
import pandas as pd
import numpy as np
import glob
import time

def read_rmcab_excel(filepath, variable_name):
    print(f"Leyendo: {filepath}")
    try:
        header_probe = pd.read_excel(filepath, header=None, nrows=10, engine='openpyxl')
        header_mask = header_probe.astype(str).apply(
            lambda col: col.str.strip().str.upper().eq('FECHA & HORA')
        )
        header_matches = header_probe.index[header_mask.any(axis=1)]
        if len(header_matches) == 0:
            raise ValueError("No se encontró la fila de encabezado 'Fecha & Hora'")

        header_row = int(header_matches[0])

        # La fila de encabezado tiene los nombres de las estaciones. Las dos
        # filas siguientes contienen variable/unidad y no son observaciones.
        df = pd.read_excel(
            filepath,
            header=header_row,
            skiprows=[header_row + 1, header_row + 2],
            engine='openpyxl'
        )
        
        # Eliminar las filas de sub-encabezados que tienen nulo en Fecha & Hora
        df = df[df.iloc[:, 0].notna()].copy()
        
        fecha_cols = [
            col for col in df.columns
            if str(col).strip().upper() == 'FECHA & HORA'
        ]
        if not fecha_cols:
            raise ValueError("No se encontró la columna 'Fecha & Hora'")
        df.rename(columns={fecha_cols[0]: 'Fecha_Hora'}, inplace=True)

        if 'Año' in df.columns:
            df.drop(columns=['Año'], inplace=True)
            
        # Limpiar columnas Unnamed
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Evitar conflicto si alguna estación casualmente se llama como la variable
        # Cambiamos temporalmente el nombre de la variable
        temp_val_name = f"VAL_{variable_name}"
        
        # Derretir (Melt)
        df_melt = pd.melt(df, id_vars=['Fecha_Hora'], var_name='Estacion', value_name=temp_val_name)
        
        # Limpieza inicial de valores usando infer_objects para evitar warnings futuros
        df_melt[temp_val_name] = df_melt[temp_val_name].replace(['Sin Data', 'Sin Dato', '----', '', ' '], np.nan)
        df_melt[temp_val_name] = pd.to_numeric(df_melt[temp_val_name], errors='coerce')
        
        # Renombrar a nombre original
        df_melt.rename(columns={temp_val_name: variable_name}, inplace=True)
        
        # Convertir a datetime
        df_melt['Fecha_Hora'] = pd.to_datetime(df_melt['Fecha_Hora'], dayfirst=True, errors='coerce')
        
        # Filtrar solo desde 2020 hasta 2025
        df_melt = df_melt[(df_melt['Fecha_Hora'] >= '2020-01-01') & (df_melt['Fecha_Hora'] < '2026-01-01')]
        
        return df_melt.dropna(subset=[variable_name])
    except Exception as e:
        print(f"Error procesando {filepath}: {e}")
        return pd.DataFrame()

def main():
    start_time = time.time()
    
    base_dir = r"Data\Raw"
    pollutants_dir = os.path.join(base_dir, r"contaminacion\PROMEDIOS HORARIOS-20260517T211058Z-3-001\PROMEDIOS HORARIOS")
    weather_dir = os.path.join(base_dir, r"variables_ambientales\PROMEDIOS HORARIOS-20260517T211053Z-3-001\PROMEDIOS HORARIOS")
    
    # Mapeo de carpetas a nombres de variables
    variables = {
        'PM2.5': (os.path.join(pollutants_dir, 'PM2.5'), 'PM2.5'),
        'PM10': (os.path.join(pollutants_dir, 'PM10'), 'PM10'),
        'NO2': (os.path.join(pollutants_dir, 'NO2'), 'NO2'),
        'CO': (os.path.join(pollutants_dir, 'CO'), 'CO'),
        'O3': (os.path.join(pollutants_dir, 'O3'), 'O3'),
        'TEMPERATURA': (os.path.join(weather_dir, 'TEMPERATURA'), 'Temperatura'),
        'HUMEDAD RELATIVA': (os.path.join(weather_dir, 'HUMEDAD RELATIVA'), 'Humedad'),
        'VELOCIDAD DEL VIENTO': (os.path.join(weather_dir, 'VELOCIDAD DEL VIENTO'), 'Viento'),
        'PRECIPITACIÓN': (os.path.join(weather_dir, 'PRECIPITACIÓN'), 'Precipitacion'),
        'RADIACIÓN SOLAR': (os.path.join(weather_dir, 'RADIACIÓN SOLAR'), 'Radiacion_Solar')
    }
    
    all_dfs = []
    
    for key, (var_folder, var_name) in variables.items():
        print(f"\n--- Procesando {var_name} ---")
        if not os.path.exists(var_folder):
            print(f"Advertencia: No se encontró la carpeta {var_folder}")
            continue
            
        excel_files = glob.glob(os.path.join(var_folder, "*.xlsx"))
        var_dfs = []
        
        for file in excel_files:
            if '2026' in os.path.basename(file):
                continue
            
            df_file = read_rmcab_excel(file, var_name)
            if not df_file.empty:
                var_dfs.append(df_file)
                
        if var_dfs:
            df_var_combined = pd.concat(var_dfs, ignore_index=True)
            df_var_combined = df_var_combined.drop_duplicates(subset=['Fecha_Hora', 'Estacion'])
            df_var_combined['Estacion'] = df_var_combined['Estacion'].str.strip().str.upper()
            all_dfs.append(df_var_combined.set_index(['Fecha_Hora', 'Estacion']))

    print("\n--- Uniendo todas las variables ---")
    if all_dfs:
        df_final = all_dfs[0]
        for df in all_dfs[1:]:
            df_final = df_final.join(df, how='outer')
            
        df_final = df_final.reset_index()
        
        # Eliminar filas donde absolutamente todo menos la estación está vacío
        vars_only = [v for k, (f, v) in variables.items()]
        df_final = df_final.dropna(subset=vars_only, how='all')
        
        print(f"\nDimensiones finales: {df_final.shape}")
        print("Muestra de datos:")
        print(df_final.head())
        
        output_path = r"Data\Processed\rmcab_merged.parquet"
        df_final.to_parquet(output_path, engine='pyarrow', index=False)
        print(f"\nGuardado exitosamente en: {output_path}")
    else:
        print("No se encontraron datos para procesar.")
        
    print(f"Tiempo total: {(time.time() - start_time)/60:.2f} minutos")

if __name__ == "__main__":
    main()
