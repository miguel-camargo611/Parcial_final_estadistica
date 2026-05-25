import pandas as pd
import numpy as np
import time

RMCAB_COORDS = {
    'BOLIVIA': (4.7358, -74.1258),
    'BOSA': (4.6250, -74.1816),
    'CARVAJAL - SEVILLANA': (4.5958, -74.1486),
    'CENTRO DE ALTO RENDIMIENTO': (4.6583, -74.0838),
    'CIUDAD BOLIVAR': (4.5772, -74.1666),
    'COLINA': (4.7371, -74.0594),
    'FONTIBON': (4.6705, -74.1415),
    'GUAYMARAL': (4.7836, -74.0436),
    'JAZMIN': (4.6083, -74.1147),
    'KENNEDY': (4.6250, -74.1438),
    'LAS FERIAS': (4.6908, -74.0824),
    'MINAMBIENTE': (4.6255, -74.0669),
    'MOVIL FONTIBON': (4.6705, -74.1415),
    'MOVIL 7MA': (4.6599, -74.0573),
    'PUENTE ARANDA': (4.6319, -74.1174),
    'SAN CRISTOBAL': (4.5722, -74.0838),
    'SUBA': (4.7613, -74.0933),
    'TUNAL': (4.5761, -74.1308),
    'USAQUEN': (4.7103, -74.0305),
    'USME': (4.5322, -74.1169)
}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 # Radio de la tierra en km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def get_nearest_station(lat, lon):
    min_dist = float('inf')
    best_station = None
    for station, (s_lat, s_lon) in RMCAB_COORDS.items():
        dist = haversine(lat, lon, s_lat, s_lon)
        if dist < min_dist:
            min_dist = dist
            best_station = station
    return best_station, min_dist

def main():
    print("Cargando eBird Shannon...")
    ebird = pd.read_parquet(r"Data\Processed\ebird_shannon.parquet")
    
    print("Cargando RMCAB Merged...")
    rmcab = pd.read_parquet(r"Data\Processed\rmcab_merged.parquet")
    
    print("Calculando estación más cercana para cada lista de eBird...")
    ebird['Nearest_Station_Tuple'] = ebird.apply(lambda row: get_nearest_station(row['LATITUDE'], row['LONGITUDE']), axis=1)
    ebird['Estacion'] = ebird['Nearest_Station_Tuple'].apply(lambda x: x[0])
    ebird['Distance_to_Station_km'] = ebird['Nearest_Station_Tuple'].apply(lambda x: x[1])
    ebird.drop(columns=['Nearest_Station_Tuple'], inplace=True)
    
    # Filtro opcional: Solo listas a menos de 5 km de la estación para asegurar representatividad
    ebird = ebird[ebird['Distance_to_Station_km'] <= 5]
    
    print("Haciendo el Match Espacio-Temporal...")
    ebird = ebird.drop(columns=['Fecha_Hora'], errors='ignore')
    ebird.rename(columns={'Fecha_Hora_Truncada': 'Fecha_Hora'}, inplace=True)
    
    # Left join
    matched = pd.merge(ebird, rmcab, on=['Fecha_Hora', 'Estacion'], how='left')
    
    output_path = r"Data\Processed\master_dataset.parquet"
    matched.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"Match completado. Guardado en {output_path}")
    print(f"Total listas cruzadas (a menos de 5km de una estación): {len(matched)}")

if __name__ == "__main__":
    main()
