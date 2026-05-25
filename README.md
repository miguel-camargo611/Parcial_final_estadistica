# Determinantes Ambientales de la Diversidad de Aves a Escala Micro-Hábitat en Bogotá

Proyecto final de Estadística Multivariada y Modelos Lineales Aplicados.

Autor: Miguel Ángel Camargo Mora  
Periodo de análisis: 2020-2025  
Unidad estadística: lista de eBird cruzada con la estación RMCAB más cercana en la hora de observación

---

## 1. Objetivo Del Proyecto

El objetivo del proyecto es estudiar cómo se relaciona la diversidad de aves observada en Bogotá con un conjunto de condiciones ambientales medidas a escala fina. Para ello se integran dos fuentes principales:

- Registros de ciencia ciudadana de eBird, usados para calcular una respuesta ecológica por evento de muestreo.
- Registros horarios de la Red de Monitoreo de Calidad del Aire de Bogotá (RMCAB), usados para caracterizar el ambiente físico y químico alrededor de cada evento.

La pregunta estadística central es:

> ¿Existen diferencias multivariadas en el ambiente físico-químico asociado a listas de alta y baja diversidad de aves, y qué variables ambientales explican mejor la variabilidad del índice de Shannon?

La estrategia general consiste en construir un dataset maestro donde cada fila represente una lista de eBird y contenga:

- respuesta ecológica: índice de Shannon;
- controles de esfuerzo: duración y distancia recorrida;
- predictores químicos: contaminantes atmosféricos;
- predictores físicos: variables meteorológicas;
- información espacial: estación RMCAB más cercana y distancia a la estación.

---

## 2. Descripción De Datos

### 2.1 Fuente eBird

La fuente ecológica proviene de eBird para Bogotá D.C. El individuo estadístico fundamental es el evento de muestreo o checklist. Cada checklist agrupa las especies observadas, sus abundancias reportadas y metadatos de esfuerzo como fecha, hora, duración y distancia recorrida.

Para evitar sesgos fuertes de calidad de observación, se aplicaron dos filtros:

1. Se conservaron únicamente listas completas, identificadas por `ALL SPECIES REPORTED == 1`.
2. Se descartaron listas con conteos no numéricos, especialmente observaciones reportadas como `X`.

Después de estos filtros y restringiendo el análisis al periodo 2020-2025, se obtuvieron 38,609 listas viables antes del filtro espacial.

Para cada lista se calculó el índice de diversidad de Shannon:

```text
H' = - sum(p_i * ln(p_i))
```

donde `p_i` es la abundancia relativa de la especie `i` dentro de la lista. Esta variable se guarda como `Shannon_Index` y actúa como respuesta principal del proyecto.

También se calculó `Riqueza_Especies`, definida como el número de especies distintas reportadas en cada lista. Aunque la variable respuesta principal es Shannon, la riqueza queda disponible como descriptor ecológico complementario.

Variables eBird conservadas en el dataset maestro:

| Variable | Descripción |
|---|---|
| `SAMPLING EVENT IDENTIFIER` | Identificador único de la lista |
| `OBSERVATION DATE` | Fecha de observación |
| `TIME OBSERVATIONS STARTED` | Hora de inicio |
| `LATITUDE`, `LONGITUDE` | Coordenadas de la lista |
| `DURATION MINUTES` | Duración de observación |
| `EFFORT DISTANCE KM` | Distancia recorrida |
| `Shannon_Index` | Índice de diversidad de Shannon |
| `Riqueza_Especies` | Número de especies observadas |

### 2.2 Fuente RMCAB

La fuente ambiental proviene de la Red de Monitoreo de Calidad del Aire de Bogotá. Los archivos originales se encuentran en formato Excel, separados por variable y año o bloque de años. El formato original es ancho: cada archivo tiene una columna temporal y una columna por estación.

Se procesaron 10 variables ambientales:

| Bloque | Variables |
|---|---|
| Contaminantes | `PM2.5`, `PM10`, `NO2`, `CO`, `O3` |
| Meteorología | `Temperatura`, `Humedad`, `Viento`, `Lluvia`, `Radiacion_Solar` |

Las variables meteorológicas se ampliaron durante el desarrollo. Inicialmente se usaban temperatura, humedad y velocidad del viento. Luego se incorporaron precipitación y radiación solar. Debido a que la precipitación horaria tenía muchos ceros, se codificó como `Lluvia`: 1 si hubo precipitación registrada en la hora y 0 si no. La radiación solar se incluyó porque captura condiciones de luz y energía ambiental que pueden afectar actividad, comportamiento y detectabilidad.

Se dejó por fuera la dirección del viento porque es una variable circular; modelarla correctamente requiere transformarla en componentes seno/coseno o trabajar con estadística circular. También se dejó por fuera presión barométrica en esta fase porque su interpretación ecológica directa es menos clara que precipitación y radiación solar.

El resultado ambiental consolidado es:

```text
Data/Processed/rmcab_merged.parquet
```

Dimensiones actuales:

```text
913,358 filas x 12 columnas
```

Las columnas corresponden a `Fecha_Hora`, `Estacion` y las 10 variables ambientales.

### 2.3 Cruce Espacio-Temporal

El dataset maestro se construyó cruzando cada lista de eBird con la estación RMCAB más cercana, usando distancia de Haversine entre las coordenadas de la lista y las coordenadas conocidas de las estaciones.

Reglas del cruce:

1. Se calcula la estación RMCAB más cercana a cada lista.
2. Se conserva solo la lista si la estación más cercana está a 5 km o menos.
3. La hora de inicio de la lista se trunca a la hora exacta (`floor('h')`).
4. Se hace un merge por `Fecha_Hora` y `Estacion`.

Resultado:

```text
Data/Processed/master_dataset.parquet
```

Dimensiones actuales:

```text
33,323 listas x 22 columnas
```

Rango temporal:

```text
2020-01-01 07:00:00 a 2025-12-31 14:00:00
```

Cobertura del dataset maestro:

| Métrica | Valor |
|---|---:|
| Listas eBird viables 2020-2025 antes del filtro espacial | 38,609 |
| Listas dentro de 5 km de una estación RMCAB | 33,323 |
| Filas con al menos una variable ambiental disponible | 31,757 |
| Filas con las 10 variables ambientales completas | 6,178 |

---

## 3. Limpieza Y Preparación

### 3.1 Limpieza de eBird

El script responsable es:

```text
Scripts/02_process_ebird.py
```

Pasos aplicados:

1. Leer el archivo tabulado original de eBird.
2. Conservar únicamente columnas necesarias para el análisis.
3. Convertir `OBSERVATION DATE` a fecha.
4. Filtrar el periodo 2020-2025.
5. Conservar listas completas (`ALL SPECIES REPORTED == 1`).
6. Eliminar listas con conteos no numéricos.
7. Calcular `Shannon_Index` por lista.
8. Calcular `Riqueza_Especies`.
9. Extraer metadatos únicos por lista.
10. Construir `Fecha_Hora`.
11. Construir `Fecha_Hora_Truncada`.
12. Guardar `Data/Processed/ebird_shannon.parquet`.

Una corrección importante realizada durante el desarrollo fue el parseo de la hora. El archivo eBird ya trae horas en formato similar a `HH:MM:SS`; por tanto, agregar manualmente `:00` generaba valores inválidos. Se corrigió usando `pd.to_timedelta` directamente sobre el texto limpio.

### 3.2 Limpieza de RMCAB

El script responsable es:

```text
Scripts/01_process_rmcab.py
```

Pasos aplicados:

1. Recorrer carpetas de contaminantes y variables meteorológicas.
2. Leer archivos Excel con encabezados heterogéneos.
3. Detectar dinámicamente la fila que contiene `Fecha & Hora`.
4. Eliminar filas de unidades y subencabezados.
5. Eliminar columna `Año` cuando aparece en archivos históricos.
6. Convertir archivos de formato ancho a formato largo mediante `melt`.
7. Estandarizar nombres de estaciones en mayúsculas.
8. Convertir textos como `Sin Data`, `Sin Dato`, `----` y espacios vacíos a `NaN`.
9. Convertir valores ambientales a numéricos.
10. Filtrar el periodo 2020-2025.
11. Unir todas las variables por `[Fecha_Hora, Estacion]` mediante outer join.
12. Guardar `Data/Processed/rmcab_merged.parquet`.

Una corrección importante fue detectar que algunos archivos Excel estaban siendo leídos desde la fila de unidades, produciendo estaciones falsas como `PPM.10` o `PPM.13`. Se solucionó detectando la fila real de encabezado y usando los nombres verdaderos de estaciones.

### 3.3 Limpieza del Dataset Maestro

El script responsable del cruce es:

```text
Scripts/03_spatial_match.py
```

Pasos aplicados:

1. Leer `ebird_shannon.parquet`.
2. Leer `rmcab_merged.parquet`.
3. Calcular estación más cercana con Haversine.
4. Guardar distancia a estación en `Distance_to_Station_km`.
5. Filtrar listas con distancia menor o igual a 5 km.
6. Usar `Fecha_Hora_Truncada` como hora de cruce.
7. Hacer merge con RMCAB por `Fecha_Hora` y `Estacion`.
8. Guardar `Data/Processed/master_dataset.parquet`.

### 3.4 Diagnóstico Inicial de Faltantes

Porcentaje de nulos en variables principales del dataset maestro:

| Variable | Nulos (%) |
|---|---:|
| PM2.5 | 11.73 |
| PM10 | 14.82 |
| NO2 | 26.73 |
| CO | 24.30 |
| O3 | 30.76 |
| Temperatura | 21.91 |
| Humedad | 38.45 |
| Viento | 25.64 |
| Lluvia | 11.68 |
| Radiacion_Solar | 55.68 |
| Shannon_Index | 0.00 |
| DURATION MINUTES | 0.06 |
| EFFORT DISTANCE KM | 27.07 |

Decisión metodológica pendiente para la etapa de limpieza:

- Para análisis multivariados que requieren matrices completas, se puede usar una base completa por subconjunto de variables.
- Para regresión, se puede evaluar imputación simple o modelos con subconjunto completo.
- Para precipitación, muchos ceros son reales, no faltantes. Por eso la matriz principal usa la codificación binaria `Lluvia`.
- Para radiación solar, el porcentaje de nulos es alto; debe decidirse si entra en todos los modelos, solo en análisis exploratorio, o en modelos alternativos.

### 3.5 Preparación Analítica Ejecutada

Se añadió y ejecutó el script:

```text
Scripts/04_prepare_analysis_dataset.py
```

Este script toma `master_dataset.parquet` y genera bases listas para EDA, inferencia, regresión y reducción dimensional. La intención es no modificar el dataset maestro original, sino derivar versiones analíticas reproducibles.

Pasos aplicados:

1. Leer `Data/Processed/master_dataset.parquet`.
2. Eliminar filas sin `Fecha_Hora` o sin `Shannon_Index`.
3. Extraer variables temporales auxiliares: `Anio`, `Mes`, `Hora`.
4. Revisar variables físicamente no negativas.
5. Convertir valores negativos físicamente imposibles a `NaN`.
6. Crear la variable `Lluvia`, codificada como 1 si `Precipitacion > 0` y 0 si `Precipitacion == 0`.
7. Crear transformaciones `log1p` auxiliares para contaminantes, precipitación cruda, radiación solar y esfuerzo.
8. Definir grupos de diversidad usando cuartiles de Shannon:
   - `Baja_Q1` si `Shannon_Index <= Q1`;
   - `Alta_Q3` si `Shannon_Index >= Q3`;
   - `Media_Q2_Q3` para valores intermedios.
9. Estandarizar predictores con `StandardScaler`.
10. Crear banderas de outliers usando `|Z| > 3`; esta regla no se aplica a `Lluvia` porque es binaria.
11. Generar bases completas para distintos análisis.
12. Guardar un resumen JSON y una tabla descriptiva en `Outputs/`.

Resultados principales de esta preparación:

| Métrica | Valor |
|---|---:|
| Filas iniciales en master | 33,323 |
| Filas en base analítica | 33,308 |
| Q1 de Shannon | 1.499 |
| Q3 de Shannon | 2.543 |
| Valores negativos de CO convertidos a `NaN` | 7 |
| Filas completas con 10 variables ambientales | 6,178 |
| Filas completas para regresión con 10 variables ambientales + esfuerzo | 5,298 |
| Filas completas sin radiación solar | 9,061 |
| Filas completas para regresión sin radiación solar | 7,276 |

Archivos generados:

```text
Data/Processed/analysis_dataset.parquet
Data/Processed/analysis_env_complete.parquet
Data/Processed/analysis_regression_complete.parquet
Data/Processed/analysis_env_reduced_no_radiation.parquet
Data/Processed/analysis_regression_reduced_no_radiation.parquet
Outputs/analysis_preparation_summary.json
Outputs/analysis_descriptive_summary.csv
```

La existencia de dos versiones completas, una con radiación solar y otra sin radiación solar, responde a una decisión metodológica importante: `Radiacion_Solar` tiene 55.66% de nulos. Incluirla permite representar mejor las condiciones de luz, pero reduce el tamaño muestral. Excluirla aumenta la potencia estadística y puede ser útil para modelos principales o de sensibilidad.

### 3.6 Decisión De Base Principal

Se decidió continuar con la base ambiental completa de 10 variables:

```text
Data/Processed/analysis_env_complete.parquet
```

Esta base tiene:

```text
6,178 filas
```

La decisión privilegia coherencia multivariada: todos los análisis ambientales principales usarán el mismo vector físico-químico completo:

```text
PM2.5, PM10, NO2, CO, O3,
Temperatura, Humedad, Viento,
Lluvia, Radiacion_Solar
```

Para regresión, cuando se incorporan controles de esfuerzo (`DURATION MINUTES` y `EFFORT DISTANCE KM`), la base completa baja a 5,298 filas. Esto ocurre principalmente por faltantes en distancia recorrida.

### 3.7 Estandarización Y Normalidad

Sí se estandarizaron los datos. La preparación analítica genera dos familias de variables estandarizadas:

```text
z_env_*
z_reg_*
```

Las variables `z_env_*` se calculan sobre la base ambiental completa de 6,178 filas y se usarán para inferencia multivariable, PCA, PPCA y Factor Analysis. Las variables `z_reg_*` se calculan sobre la base completa de regresión de 5,298 filas e incluyen ambiente + esfuerzo.

Es importante distinguir dos operaciones:

- **Transformar o normalizar la forma de la distribución:** aplicar, por ejemplo, `log1p(x)` para reducir asimetría.
- **Estandarizar:** convertir a Z-score, es decir, media 0 y desviación estándar 1.

La estandarización no vuelve normal una variable; solo la pone en una escala comparable. Por eso primero se aplicaron transformaciones `log1p` a variables con colas fuertes y después se estandarizaron las variables transformadas.

Variables con transformación `log1p` conservadas como diagnóstico auxiliar:

```text
PM2.5, PM10, NO2, CO, O3,
Radiacion_Solar,
DURATION MINUTES, EFFORT DISTANCE KM
```

Variables usadas en la matriz principal:

```text
PM2.5, PM10, NO2, CO, O3,
Temperatura, Humedad, Viento,
Lluvia, Radiacion_Solar
```

Se ejecutó un diagnóstico univariado de distribución sobre la base ambiental completa:

```text
Scripts/05_distribution_diagnostics.py
Outputs/distribution_diagnostics_env_complete.csv
```

Resultado general:

- Las pruebas formales de normalidad (Shapiro-Wilk sobre muestra de máximo 5,000 observaciones y D'Agostino) rechazan normalidad para todas las variables.
- Este rechazo debe interpretarse con cautela porque el tamaño muestral es alto (`n = 6,178`), lo que hace que desviaciones pequeñas frente a la normalidad produzcan valores-p muy bajos.
- Las transformaciones `log1p` reducen sustancialmente la asimetría de PM2.5, PM10, NO2, CO, O3 y Radiacion_Solar, pero no logran normalidad formal.
- La precipitación cruda continúa siendo muy asimétrica aun con `log1p`, porque es una variable con muchos ceros reales. Por esta razón se reemplazó en la matriz principal por `Lluvia`.
- Temperatura es la variable más cercana a simetría, aunque tampoco pasa normalidad formal.

Diagnóstico resumido de asimetría:

| Variable | Asimetría original | Escala usada en modelos | Asimetría en escala usada |
|---|---:|---|---:|
| PM2.5 | 1.227 | `log1p_PM2_5` | -0.370 |
| PM10 | 1.136 | `log1p_PM10` | -0.437 |
| NO2 | 1.008 | `log1p_NO2` | -0.635 |
| CO | 1.602 | `log1p_CO` | 0.853 |
| O3 | 1.423 | `log1p_O3` | -0.101 |
| Temperatura | 0.032 | `Temperatura` | 0.032 |
| Humedad | -0.466 | `Humedad` | -0.466 |
| Viento | 1.156 | `Viento` | 1.156 |
| Lluvia | 3.910 | `Lluvia` | 3.910 |
| Radiacion_Solar | 1.070 | `log1p_Radiacion_Solar` | -0.718 |

Decisión metodológica actual:

- Los análisis principales usarán variables ambientales originales, no transformadas, para preservar interpretabilidad física.
- Para métodos sensibles a escala, como PCA, PPCA y FA, se usarán variables originales estandarizadas (`z_env_*`).
- Para regresión lineal también se usarán variables originales estandarizadas (`z_reg_*`) cuando se comparen coeficientes entre predictores. Esto evita que variables con unidades grandes dominen la magnitud numérica del coeficiente. La interpretación sustantiva se hará en unidades originales cuando sea útil.
- La normalidad multivariada no se asumirá. En lugar de depender únicamente de Hotelling T2, la inferencia multivariada principal se apoyará en una alternativa no paramétrica/permutacional para comparar grupos de baja y alta diversidad.
- Hotelling T2 puede reportarse como referencia clásica si se desea, pero no será la única evidencia inferencial.
- `Lluvia` se conserva por relevancia ecológica e interpretabilidad directa. La precipitación cruda queda como variable auxiliar descriptiva.

---

## 4. EDA

El Análisis Exploratorio de Datos debe responder tres preguntas:

1. ¿Cómo se distribuye la diversidad de aves medida por Shannon?
2. ¿Cómo se distribuyen los contaminantes y variables meteorológicas?
3. ¿Qué relaciones iniciales aparecen entre diversidad, esfuerzo y ambiente?

### 4.1 Estadísticas Descriptivas Iniciales

Resumen actual del dataset maestro:

| Variable | Media | Mediana | Mínimo | Máximo |
|---|---:|---:|---:|---:|
| `Shannon_Index` | 1.966 | 2.042 | 0.000 | 3.912 |
| `Riqueza_Especies` | 13.344 | 11.000 | 1.000 | 65.000 |
| `Distance_to_Station_km` | 2.017 | 2.031 | 0.020 | 4.990 |
| `PM2.5` | 16.325 | 14.000 | 0.000 | 120.000 |
| `PM10` | 31.410 | 26.300 | 0.100 | 241.000 |
| `NO2` | 16.001 | 14.800 | 0.000 | 99.400 |
| `CO` | 0.655 | 0.530 | -0.030 | 5.980 |
| `O3` | 11.748 | 8.200 | 0.000 | 95.300 |
| `Temperatura` | 14.792 | 14.200 | 3.200 | 27.300 |
| `Humedad` | 67.607 | 70.000 | 11.000 | 100.000 |
| `Viento` | 1.415 | 1.100 | 0.000 | 7.900 |
| `Lluvia` | 0.055 | 0.000 | 0.000 | 1.000 |
| `Precipitacion` auxiliar | 0.029 | 0.000 | 0.000 | 18.600 |
| `Radiacion_Solar` | 198.858 | 118.000 | 0.000 | 1169.000 |

Lecturas iniciales:

- Shannon tiene una media cercana a 2, con valores máximos alrededor de 3.9.
- `Lluvia` muestra que cerca de 5.5% de las observaciones completas tienen precipitación horaria positiva.
- Radiación solar tiene muchos valores en 0 o bajos, consistente con observaciones nocturnas o de baja luz, pero también tiene alto porcentaje de faltantes.
- PM10 y PM2.5 presentan colas largas, por lo que probablemente requieran transformación logarítmica para algunos análisis.

### 4.2 Visualizaciones A Construir

Pendientes para notebooks:

1. Histograma y KDE de `Shannon_Index`.
2. Histogramas de contaminantes.
3. Histogramas de variables meteorológicas.
4. Boxplots por estación.
5. Scatterplots de Shannon contra esfuerzo.
6. Scatterplots de Shannon contra contaminantes.
7. Heatmap de correlación.
8. Matriz de dispersión reducida con variables clave.

### 4.3 Correlaciones

La matriz de correlación debe incluir:

- Shannon;
- riqueza;
- esfuerzo;
- distancia a estación;
- contaminantes;
- meteorología.

Se debe interpretar con cuidado porque correlación no implica causalidad y porque eBird introduce sesgos de esfuerzo, detectabilidad y selección espacial.

### 4.4 Insumo Actual Para EDA

La tabla descriptiva inicial fue guardada en:

```text
Outputs/analysis_descriptive_summary.csv
```

Esta tabla resume las variables ambientales, los controles de esfuerzo, Shannon y riqueza. Será la base para construir la primera tabla formal de EDA del informe.

Además, `analysis_dataset.parquet` contiene:

- variables originales;
- transformaciones `log1p`;
- variables estandarizadas con prefijo `z_`;
- grupo de diversidad por cuartiles;
- banderas de outliers por Z-score.

Con esto el EDA puede avanzar directamente a gráficos y correlaciones sin repetir limpieza.

### 4.5 EDA Ejecutado

Se añadió y ejecutó el script:

```text
Scripts/06_eda.py
```

El EDA se realizó sobre la base principal:

```text
Data/Processed/analysis_env_complete.parquet
```

Esta base contiene 6,178 filas con las 10 variables ambientales completas. Se generaron tablas descriptivas, matrices de correlación y figuras exploratorias.

Archivos generados:

```text
Outputs/eda_descriptive_env_complete.csv
Outputs/eda_q1_q3_group_summary.csv
Outputs/eda_corr_pearson.csv
Outputs/eda_corr_spearman.csv
Outputs/eda_top_correlations_with_shannon.csv
Figures/EDA/01_shannon_distribution.png
Figures/EDA/02_environment_distributions_original.png
Figures/EDA/03_pearson_correlation_heatmap.png
Figures/EDA/04_shannon_vs_environment_scatter.png
Figures/EDA/05_q1_q3_environment_boxplots_z.png
```

Correlaciones de Pearson más relevantes con `Shannon_Index`:

| Variable | Correlación con Shannon |
|---|---:|
| `Riqueza_Especies` | 0.846 |
| `Temperatura` | -0.350 |
| `Humedad` | 0.318 |
| `O3` | -0.297 |
| `Viento` | -0.276 |
| `Radiacion_Solar` | -0.273 |
| `NO2` | 0.087 |
| `CO` | 0.086 |
| `Lluvia` | -0.031 |
| `PM10` | -0.020 |
| `PM2.5` | 0.010 |

Comparación exploratoria entre baja y alta diversidad:

| Variable | Baja diversidad Q1 | Alta diversidad Q3 | Lectura inicial |
|---|---:|---:|---|
| `Shannon_Index` medio | 1.028 | 2.823 | Separación esperada por construcción de grupos |
| `Riqueza_Especies` media | 4.847 | 24.539 | Alta diversidad también implica mayor riqueza |
| `Temperatura` media | 16.805 | 12.888 | Alta diversidad aparece en condiciones más frías |
| `Humedad` media | 61.230 | 75.466 | Alta diversidad aparece en condiciones más húmedas |
| `O3` medio | 17.282 | 7.216 | Alta diversidad aparece con menor ozono horario |
| `Viento` medio | 1.677 | 0.978 | Alta diversidad aparece con menor viento |
| `Radiacion_Solar` media | 289.869 | 125.038 | Alta diversidad aparece con menor radiación |
| `PM2.5` media | 14.852 | 15.145 | Diferencia exploratoria pequeña |
| `PM10` media | 29.491 | 28.856 | Diferencia exploratoria pequeña |

Interpretación preliminar:

> Las listas de alta diversidad se concentran, de forma exploratoria, en horas o micro-hábitats más fríos, húmedos, con menor viento, menor radiación solar y menor ozono. En contraste, PM2.5 y PM10 no muestran diferencias lineales fuertes frente a Shannon en esta primera aproximación. Estos patrones no deben interpretarse causalmente todavía, porque pueden estar confundidos por hora del día, ubicación, estación, esfuerzo de observación y selección espacial de los observadores.

El siguiente paso será pasar de esta lectura descriptiva a análisis más estructurados:

- distribuciones conjuntas bivariadas;
- prueba permutacional entre baja y alta diversidad;
- regresión lineal con controles de esfuerzo;
- PCA/PPCA/FA para estructura ambiental latente.

---

## 5. Inferencia Multivariable

La inferencia multivariable se centrará en comparar ambientes asociados a listas de baja y alta diversidad.

### 5.1 Definición de Grupos

Se definirán grupos usando cuartiles de `Shannon_Index`:

- Baja diversidad: listas en Q1.
- Alta diversidad: listas en Q3 o superior, según la definición operacional que se adopte en el notebook.

La idea es evitar una comparación débil alrededor de la mediana. Comparar extremos de la distribución aumenta el contraste ecológico entre grupos.

### 5.2 Variables Para La Comparación

Para la inferencia multivariada se usará el vector ambiental original estandarizado:

```text
PM2.5, PM10, NO2, CO, O3,
Temperatura, Humedad, Viento,
Lluvia, Radiacion_Solar
```

Dependiendo de la cobertura, se puede construir:

- Modelo ambiental completo con las 10 variables.
- Modelo ambiental reducido sin `Radiacion_Solar` si la pérdida de muestra es demasiado fuerte.

### 5.3 Supuestos Y Enfoque No Paramétrico

Antes de aplicar pruebas clásicas como Hotelling T2 se deben revisar:

1. Normalidad multivariada, usando Mardia.
2. Homogeneidad de matrices de covarianza, usando Box's M.
3. Tamaño muestral por grupo.
4. Ausencia de colinealidad extrema.

Como las pruebas univariadas ya sugieren desviaciones claras de normalidad, la ruta principal será una prueba multivariada no paramétrica/permutacional. Esta estrategia permite comparar los grupos de baja y alta diversidad sin exigir normalidad multivariada estricta.

La prueba recomendada es PERMANOVA o una prueba permutacional de distancia entre centroides sobre la matriz ambiental estandarizada. Esta prueba evalúa si la separación entre grupos es mayor que la esperada bajo reasignaciones aleatorias de las etiquetas de grupo.

### 5.4 Resultado Esperado

El resultado principal será determinar si la composición ambiental multivariada de listas de alta diversidad difiere significativamente de la composición ambiental de listas de baja diversidad.

En lenguaje de informe:

> Se evaluó si las condiciones ambientales multivariadas difieren entre listas de baja y alta diversidad. Para ello se construyeron grupos a partir de los cuartiles del índice de Shannon y se comparó la matriz ambiental estandarizada mediante una prueba permutacional, evitando depender del supuesto estricto de normalidad multivariada.

---

## 6. Regresión Lineal Múltiple

La regresión busca explicar variación en `Shannon_Index` usando variables ambientales y controles de esfuerzo.

### 6.1 Modelo Base

La especificación conceptual es:

```text
Shannon_Index ~ contaminantes + meteorología + duración + distancia recorrida
```

Variables predictoras candidatas en escala original:

```text
PM2.5, PM10, NO2, CO, O3,
Temperatura, Humedad, Viento,
Lluvia, Radiacion_Solar,
DURATION MINUTES, EFFORT DISTANCE KM
```

Todas las variables continuas predictoras pueden estandarizarse para comparar magnitudes de coeficientes, pero la escala base del análisis se mantiene en unidades originales.

### 6.2 Controles De Esfuerzo

Los controles de esfuerzo son cruciales porque eBird no es un muestreo experimental balanceado. Una lista más larga o con mayor distancia recorrida puede observar más especies y modificar el índice de Shannon. Por eso `DURATION MINUTES` y `EFFORT DISTANCE KM` entran en la regresión aunque no entran en PCA/FA ambiental.

### 6.3 Diagnósticos

Se deben reportar:

1. VIF para multicolinealidad.
2. Breusch-Pagan para homocedasticidad.
3. QQ-plot y Shapiro-Wilk para normalidad de residuales.
4. Durbin-Watson para independencia aproximada.
5. Residuales vs ajustados.

### 6.4 Interpretación

Cuando se use la versión estandarizada, cada coeficiente representa el cambio esperado en Shannon ante un aumento de una desviación estándar en el predictor, manteniendo constantes las demás variables.

Esto permite comparar magnitudes:

- efecto relativo de contaminantes;
- efecto relativo de clima;
- efecto relativo de esfuerzo de observación.

---

## 7. PCA, PPCA Y Factor Analysis

Esta sección estudia la estructura latente del ambiente multivariado, no la respuesta directamente.

### 7.1 Variables Incluidas

Para PCA, PPCA y Factor Analysis se usarán únicamente variables ambientales originales estandarizadas:

```text
PM2.5, PM10, NO2, CO, O3,
Temperatura, Humedad, Viento,
Lluvia, Radiacion_Solar
```

No se incluyen duración ni distancia recorrida porque son variables de esfuerzo humano, no dimensiones físicas del ambiente.

### 7.2 PCA

El PCA permitirá resumir el ambiente en componentes ortogonales. Se reportará:

1. Varianza explicada por componente.
2. Scree plot.
3. Cargas de variables en PC1 y PC2.
4. Biplot de observaciones.
5. Biplot coloreado por baja/alta diversidad.

Interpretación esperada:

- Un componente puede capturar gradientes de contaminación.
- Otro componente puede capturar gradientes meteorológicos.
- El coloreo por diversidad permitirá observar si las listas de alta diversidad se concentran en zonas particulares del espacio ambiental.

### 7.3 PPCA

El PPCA se usará como extensión probabilística del PCA. Su aporte principal será estimar una varianza de ruido asociada a la representación latente. En contexto RMCAB, esto puede interpretarse como parte de la variabilidad ambiental no capturada por los componentes principales o asociada a ruido de medición/sensores.

### 7.4 Factor Analysis

El Factor Analysis busca factores latentes interpretables. Se aplicará rotación Varimax para facilitar que cada variable cargue fuertemente en pocos factores.

Posibles factores esperados:

- Factor de material particulado: PM2.5 y PM10.
- Factor de gases contaminantes: NO2, CO, O3.
- Factor meteorológico: temperatura, humedad, viento, lluvia y radiación.

### 7.5 Comparación PCA, PPCA Y FA

La comparación final debe enfatizar:

| Método | Propósito | Salida clave |
|---|---|---|
| PCA | Reducir dimensionalidad | Componentes y varianza explicada |
| PPCA | Modelo probabilístico con ruido | Componentes y varianza residual |
| FA | Identificar factores latentes interpretables | Cargas rotadas |

---

## 8. Ejecución Reproducible

Orden actual de ejecución:

```powershell
python Scripts\01_process_rmcab.py
python Scripts\02_process_ebird.py
python Scripts\03_spatial_match.py
python Scripts\04_prepare_analysis_dataset.py
python Scripts\05_distribution_diagnostics.py
python Scripts\06_eda.py
```

Archivos generados:

```text
Data/Processed/rmcab_merged.parquet
Data/Processed/ebird_shannon.parquet
Data/Processed/master_dataset.parquet
Data/Processed/analysis_dataset.parquet
Data/Processed/analysis_env_complete.parquet
Data/Processed/analysis_regression_complete.parquet
Data/Processed/analysis_env_reduced_no_radiation.parquet
Data/Processed/analysis_regression_reduced_no_radiation.parquet
Outputs/distribution_diagnostics_env_complete.csv
Outputs/eda_descriptive_env_complete.csv
Outputs/eda_q1_q3_group_summary.csv
Outputs/eda_corr_pearson.csv
Outputs/eda_corr_spearman.csv
Outputs/eda_top_correlations_with_shannon.csv
Figures/EDA/*.png
```

Próximo paso:

```text
Construir EDA:
- tabla descriptiva interpretada;
- histogramas/KDE;
- matriz de correlación;
- scatterplots clave;
- evaluación visual de outliers y transformaciones.
```

---

## 9. Estado Actual Del Proyecto

Completado:

- Estructura de carpetas.
- Procesamiento de RMCAB.
- Procesamiento de eBird.
- Cálculo de Shannon.
- Cruce espacial por estación más cercana.
- Cruce temporal por hora.
- Incorporación de lluvia binaria y radiación solar.
- Dataset maestro generado.
- Base analítica preparada.
- Transformaciones logarítmicas generadas.
- Variables estandarizadas generadas.
- Grupos Q1/Q3 de Shannon generados.
- Reporte descriptivo inicial generado.
- EDA principal generado sobre 6,178 filas completas.
- Figuras exploratorias generadas.

En curso:

- Construcción de notebooks de análisis.

Pendiente:

- Inferencia multivariable.
- Regresión lineal múltiple.
- PCA, PPCA y Factor Analysis.
- Redacción final consolidada.
