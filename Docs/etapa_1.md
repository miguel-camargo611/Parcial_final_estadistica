# Etapa 1 — Ingesta y Cruce Espacio-Temporal
## Estado del Proyecto: Fase de Ingesta y Match Completada ✅

> **Fecha de cierre de etapa:** 22 de mayo de 2026  
> **Duración:** ~1 sesión de trabajo  
> **Estado global del proyecto:** 🟡 En progreso — Etapa 1 terminada, listo para limpieza y análisis

---

## ✅ Lo que hicimos en esta etapa

### 1. Planificación y Diseño del Proyecto
- Discutimos extensamente el enfoque metodológico y acordamos trabajar a nivel de **lista de eBird** (evento de muestreo individual) como unidad estadística, en lugar de promediar por día o por ciudad. Este fue un punto de diseño crucial: evita el *effort bias* (si una persona observó 6 horas y otra 15 minutos, no podemos promediar sus listas).
- Definimos el **Índice de Shannon ($H'$)** como variable respuesta, calculado por lista.
- Acordamos el umbral de **Q1 vs Q3 del Shannon** para crear los grupos de Alta/Baja Diversidad (más sólido que la mediana y con respaldo en ecología urbana).
- Acordamos estandarizar **absolutamente todos** los predictores con `StandardScaler` (contaminantes + clima + esfuerzo) para hacer los coeficientes $\beta$ comparables entre sí.

### 2. Estructura del Repositorio
Se creó la siguiente arquitectura de carpetas y se reorganizaron los datos originales:
```
Parcial FInal Estadistica/
├── Data/
│   ├── Raw/          ← contaminacion, variables_ambientales, ebird (originales intactos)
│   └── Processed/    ← archivos .parquet generados por los scripts
├── Scripts/          ← 01, 02, 03 completados
├── Notebooks/        ← pendientes
├── Figures/          ← pendiente
├── Outputs/          ← pendiente
└── Docs/             ← informe final pendiente
```

### 3. Script 01 — Procesamiento RMCAB ✅
`Scripts/01_process_rmcab.py`

- Iteró sobre **10 variables** ambientales: PM2.5, PM10, NO₂, CO, O₃, Temperatura, Humedad, Velocidad del Viento, Precipitación y Radiación Solar.
- Procesó archivos Excel de formato "ancho" (una columna por estación) aplicando un **Melt** para convertirlos a formato largo: `[Fecha_Hora, Estacion, Valor]`.
- Filtró el período **2020–2025** y limpió etiquetas de nulo (`Sin Dato`, `----`).
- Realizó un **Outer Join** por índice `[Fecha_Hora, Estacion]` para unir las 10 variables en un único DataFrame.
- **Resultado guardado:** `Data/Processed/rmcab_merged.parquet`
  - Dimensiones: **913,358 filas × 12 columnas** (Fecha_Hora, Estacion + 10 variables).
  - Tiempo de procesamiento: ~6 minutos (archivos Excel de gran tamaño).

> ✅ **Corrección aplicada:** Se ajustó la lectura de encabezados para usar los nombres reales de estaciones en lugar de códigos de unidad como `PPM.10` o `PPM.13`.

### 4. Script 02 — Cálculo del Índice de Shannon (eBird) ✅
`Scripts/02_process_ebird.py`

- Leyó el archivo completo de eBird: **770,493 observaciones** originales.
- Aplicó el filtro doble:
  1. `ALL SPECIES REPORTED == 1` → Solo listas completas.
  2. `OBSERVATION COUNT` 100% numérico → Sin conteos aproximados ("X").
- **Resultado: 38,609 listas viables** para el período 2020–2025.
- Calculó el Índice de Shannon: $H' = -\sum p_i \ln(p_i)$ por lista.
- Extrajo metadatos por lista: Fecha, Hora, Latitud, Longitud, Duración, Distancia.
- **Resultado guardado:** `Data/Processed/ebird_shannon.parquet`

### 5. Script 03 — Match Espacio-Temporal ✅
`Scripts/03_spatial_match.py`

- Calcula la estación RMCAB más cercana a cada lista de eBird usando la **fórmula geodésica de Haversine**.
- Filtra listas a un radio máximo de **5 km** de la estación.
- Hace el cruce temporal truncando la hora de inicio de la lista con el registro horario de la RMCAB.
- **Resultado guardado:** `Data/Processed/master_dataset.parquet`
  - Dimensiones: **33,323 listas × 22 columnas** dentro de un radio máximo de 5 km.
  - Filas con al menos una variable ambiental disponible: **31,757**.
  - Filas con las 10 variables ambientales completas: **6,179**.

---

## ⏳ Lo que falta por hacer

### Próximo inmediato (Etapa 2)
| Tarea | Archivo | Estado |
|---|---|---|
| Verificar calidad del match (cuántas listas tienen datos RMCAB completos) | `master_dataset.parquet` | ✅ Completado |
| Notebook 01 — Limpieza y detección de outliers | `Notebooks/01_Limpieza.ipynb` | ⬜ |
| Notebook 02 — EDA y Scatter Matrix | `Notebooks/02_EDA.ipynb` | ⬜ |
| Notebook 03 — Distribuciones Conjuntas | `Notebooks/03_Distribuciones.ipynb` | ⬜ |
| Notebook 04 — Inferencia Hotelling T² | `Notebooks/04_Inferencia.ipynb` | ⬜ |
| Notebook 05 — Regresión Múltiple | `Notebooks/05_Regresion.ipynb` | ⬜ |
| Notebook 06 — PCA, PPCA y FA | `Notebooks/06_PCA_PPCA_FA.ipynb` | ⬜ |

---

## 🧭 Hacia dónde vamos — Etapa 2

El dataset maestro (`master_dataset.parquet`) será el punto de partida de **todo el análisis estadístico**. El flujo de trabajo será:

```
master_dataset.parquet
        │
        ▼
┌──────────────────────┐
│ Notebook 01          │   → Outliers (Z > 3), nulos, log-transform, estandarización
│ Limpieza             │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Notebook 02          │   → Scatter matrix, heatmap correlación, descriptivas
│ EDA                  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Notebook 03          │   → KDE 2D, Normal Bivariada, Condicionales f(H'|PM2.5)
│ Distribuciones       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Notebook 04          │   → Mardia, Box's M, Hotelling T² (Q1 vs Q3 Shannon)
│ Inferencia           │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Notebook 05          │   → OLS, VIF, Breusch-Pagan, Shapiro, Durbin-Watson
│ Regresión            │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Notebook 06          │   → PCA Biplot (coloreado Q1/Q3), PPCA, Varimax FA
│ PCA / PPCA / FA      │
└──────────────────────┘
```

---

## 📌 Decisiones de Diseño Tomadas (para no olvidar)

| Decisión | Justificación |
|---|---|
| Unidad estadística = Lista de eBird | Evita el *effort bias* del conteo de aves |
| Variable respuesta = Shannon $H'$ | Continua, más informativa que la riqueza entera |
| Umbral Alta/Baja = Q1 vs Q3 | Contraste ecológico fuerte, estándar en ecología urbana |
| Estandarización de TODO | Comparar $\beta$ de minutos vs $\mu g/m^3$ es imposible sin esto |
| Esfuerzo solo en Regresión | En PCA/FA arruinaría la interpretación física de los factores |
| Biplot coloreado por diversidad | Conecta visualmente la reducción dimensional con la ecología |
| Solo 1 gráfico 3D | Para la Normal Bivariada teórica; los demás en 2D (más legibles) |
