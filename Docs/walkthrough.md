# Borrador del Informe Final
*Este documento se actualizará en vivo a medida que procesamos los datos y ejecutamos los modelos. Puedes usar el texto generado aquí directamente para tu PDF final.*

## 2. Descripción de Datos

Este estudio integra tres fuentes de información heterogéneas, cruzadas mediante una resolución espacio-temporal fina (a nivel de hora y coordenada geográfica) en la ciudad de Bogotá D.C., durante el período 2020–2025.

1. **Variables de Respuesta Ecológica (eBird):** 
   Se empleó la base de datos de ciencia ciudadana de eBird para Bogotá. El individuo estadístico fundamental es el "Evento de Muestreo" (Checklist). Tras un filtrado riguroso, seleccionamos exclusivamente listas catalogadas como "Completas" (`ALL SPECIES REPORTED == 1`) y con conteos poblacionales 100% numéricos (sin "X") durante 2020–2025. Esto arrojó un total de **38,609 eventos de muestreo viables** antes del filtro espacial y **33,323 listas cruzadas** dentro de un radio máximo de 5 km de una estación RMCAB. A partir de la abundancia relativa de las especies observadas en cada lista, se calculó el **Índice de Diversidad de Shannon ($H'$)**, el cual fungirá como nuestra variable respuesta ($Y$).

2. **Variables de Control de Esfuerzo (eBird):**
   Para aislar el efecto ambiental de los sesgos introducidos por la actividad humana, se extrajeron variables indicadoras del esfuerzo de muestreo por lista: *Duración de observación (minutos)* y *Distancia recorrida (km)*.

3. **Ambiente Multivariado Físico y Químico (RMCAB):**
   Se consolidó el registro histórico de la Red de Monitoreo de Calidad del Aire de Bogotá (RMCAB), procesando cientos de archivos en formato amplio (wide) para obtener una base de datos horaria por estación. El vector ambiental ($X$) quedó compuesto por:
   * **Contaminantes químicos:** PM2.5, PM10, NO₂, CO y O₃ ($\mu g/m^3$ y $mg/m^3$).
   * **Variables meteorológicas (Clima):** Temperatura ($°C$), Humedad Relativa ($\%$), Velocidad del Viento ($m/s$), Lluvia horaria codificada como variable binaria y Radiación Solar.

**Algoritmo de Cruce Espacio-Temporal:** Cada lista de eBird fue proyectada espacialmente. Mediante la fórmula de distancia de Haversine, se localizó la estación de monitoreo de la RMCAB más cercana (a un radio máximo de 5 km). Posteriormente, se cruzó el índice de biodiversidad de dicha lista con las mediciones exactas de contaminación y clima reportadas por la estación en la hora precisa en la que inició la observación de aves. Este cruce permite inferir las condiciones ambientales precisas ("micro-hábitats") a las que estaban sometidas las aves.

## 3. Limpieza y Preparación

La limpieza y curación de datos ambientales representó un desafío debido a la heterogeneidad de los sensores y a las fallas inherentes a las redes de monitoreo.

1. **Unificación y Fusión (Melt):** Las bases de datos originales presentaban un formato ancho (una columna por estación). Se realizó un "melting" iterativo para todas las variables, resultando en una estructura larga indexada por `[Fecha_Hora, Estación]`.
2. **Tratamiento de Valores Faltantes (Nulos):** Los registros etiquetados como "Sin Dato" o "-----" fueron casteados a nulos (`NaN`). Dada la resolución a nivel de evento de muestreo, las listas de aves que, al ser cruzadas con la estación más cercana, presentaban datos ambientales nulos, fueron excluidas del análisis para asegurar la integridad de la matriz multivariada.
3. **Estandarización de Estaciones:** Se normalizaron los identificadores de texto de las ~20 estaciones activas para permitir un "Outer Join" entre las 10 variables ambientales.

## 4. EDA (Análisis Exploratorio)

El EDA se realizó sobre la base completa de 10 variables ambientales (`analysis_env_complete.parquet`, 6,178 filas). En esta etapa se conservaron las variables en escala original para interpretación y se usaron versiones estandarizadas únicamente para comparar perfiles ambientales entre grupos.

Las correlaciones lineales más fuertes con `Shannon_Index` fueron: riqueza de especies (`r = 0.846`), temperatura (`r = -0.350`), humedad (`r = 0.318`), ozono (`r = -0.297`), viento (`r = -0.276`) y radiación solar (`r = -0.273`). PM2.5 y PM10 presentaron correlaciones lineales débiles con Shannon en esta primera aproximación.

Al comparar los grupos extremos de diversidad, las listas de alta diversidad presentaron condiciones promedio más frías, húmedas, con menor viento, menor radiación solar y menor ozono. Esta lectura se interpreta como exploratoria y no causal, debido a posibles efectos de hora del día, ubicación, esfuerzo de muestreo y selección espacial de los observadores.

## 5. Distribuciones Conjuntas

Se construyó el Notebook 02:

```text
Notebooks/02_distribuciones_conjuntas.ipynb
```

También se dejó una versión reproducible como script:

```text
Scripts/08_joint_distributions.py
```

El par bivariado principal fue `Temperatura` y `Humedad`, ambas estandarizadas como `z_env_Temperatura` y `z_env_Humedad`. Se escogió este par porque son variables continuas, tienen asociación exploratoria importante con Shannon y muestran una relación conjunta clara.

Salidas generadas:

```text
Figures/Joint_Distributions/01_kde2d_temperatura_humedad.png
Figures/Joint_Distributions/02_elipses_densidad_temperatura_humedad.png
Figures/Joint_Distributions/03_normal_bivariada_temperatura_humedad.png
Figures/Joint_Distributions/04_shannon_condicional_contaminacion.png
Outputs/joint_bivariate_summary.csv
Outputs/joint_conditional_pollution_summary.csv
Outputs/joint_pollution_thresholds.csv
```

La KDE 2D muestra que la relación entre temperatura y humedad no es perfectamente elíptica. La normal bivariada ajustada captura la tendencia negativa general, pero suaviza una estructura observada más curvada y con concentraciones locales. Esto refuerza la decisión de no depender exclusivamente de supuestos gaussianos estrictos.

Para las distribuciones condicionales se construyó un índice de contaminación como el promedio de `PM2.5`, `PM10`, `NO2`, `CO` y `O3` estandarizados. Los grupos extremos se definieron por cuartiles del índice:

```text
Q1 contaminación = -0.483
Q3 contaminación = 0.350
```

Las distribuciones de Shannon bajo baja y alta contaminación se solapan fuertemente. El promedio de Shannon fue 2.099 en contaminación baja y 2.059 en contaminación alta. Esta diferencia descriptiva es pequeña y debe evaluarse formalmente en la siguiente fase con comparación multivariada Q1 vs Q3 de diversidad.

## 6. Inferencia Multivariable

Se construyó el Notebook 03:

```text
Notebooks/03_inferencia_multivariable.ipynb
```

También se dejó una versión reproducible como script:

```text
Scripts/09_multivariate_inference_assumptions.py
```

En esta etapa se evaluaron los supuestos clásicos usando solo las 9 variables ambientales continuas estandarizadas:

```text
PM2.5, PM10, NO2, CO, O3,
Temperatura, Humedad, Viento, Radiacion_Solar
```

`Lluvia` no se incluyó en estas pruebas porque está codificada como variable binaria. Esta decisión permite evaluar normalidad multivariada, homogeneidad de covarianzas y Hotelling T2 sobre una matriz continua.

Los grupos comparados fueron:

```text
Baja_Q1: 1,075 listas
Alta_Q3: 1,760 listas
```

Resultados principales:

- Mardia rechazó normalidad multivariada tanto en la muestra completa como en los grupos extremos de diversidad (`p < 0.001` para asimetría y curtosis).
- Box's M rechazó igualdad de matrices de covarianza entre `Baja_Q1` y `Alta_Q3` (`chi2 corregido = 1466.055`, `gl = 45`, `p < 0.001`).
- Hotelling T2 detectó diferencia significativa de centroides ambientales (`T2 = 885.450`, `F = 98.106`, `gl = 9, 2825`, `p < 0.001`), pero debe reportarse como referencia clásica porque sus supuestos no se cumplen.
- La prueba permutacional de distancia entre centroides, con 9,999 permutaciones, también rechazó igualdad de centroides (`distancia observada = 1.921`, `p = 0.0001`). Esta prueba se interpreta como la evidencia inferencial principal porque no depende de normalidad multivariada ni de homogeneidad estricta de covarianzas.

Las mayores diferencias medias estandarizadas entre alta y baja diversidad fueron:

| Variable | Alta - Baja |
|---|---:|
| Temperatura | -0.972 |
| Humedad | 0.881 |
| O3 | -0.838 |
| Viento | -0.770 |
| Radiacion_Solar | -0.751 |
| NO2 | 0.235 |

Esto indica que las listas de alta diversidad se asocian exploratoriamente con ambientes más fríos, más húmedos, con menor ozono, menor viento y menor radiación solar. Debido al rechazo de normalidad multivariada y homocedasticidad, Hotelling T2 queda como referencia clásica y la prueba permutacional queda como comparación inferencial principal.

Salidas generadas:

```text
Outputs/multivar_mardia_continuous.csv
Outputs/multivar_box_m_continuous.csv
Outputs/multivar_hotelling_continuous.csv
Outputs/multivar_permutation_centroids_continuous.csv
Outputs/multivar_permutation_distances_continuous.csv
Outputs/multivar_group_mean_differences_continuous.csv
Figures/Multivariate_Inference/01_group_mean_profile_continuous.png
Figures/Multivariate_Inference/02_centroid_confidence_ellipses_temp_humidity.png
Figures/Multivariate_Inference/03_permutation_centroid_distance_continuous.png
```

## 7. Regresión Lineal Múltiple

Se construyó el Notebook 04:

```text
Notebooks/04_regresion_lineal_multiple.ipynb
```

También se dejó una versión reproducible como script:

```text
Scripts/10_multiple_regression.py
```

El modelo base fue:

```text
Shannon_Index ~ Contaminantes_Z + Clima_Z + Duracion_Z + Distancia_Z
```

La base usada fue:

```text
Data/Processed/analysis_regression_complete.parquet
```

El modelo se ajustó con 5,298 filas completas. Los predictores fueron las 10 variables ambientales estandarizadas (`PM2.5`, `PM10`, `NO2`, `CO`, `O3`, `Temperatura`, `Humedad`, `Viento`, `Lluvia`, `Radiacion_Solar`) y dos controles de esfuerzo (`DURATION MINUTES`, `EFFORT DISTANCE KM`), también estandarizados.

Resultados principales del ajuste:

```text
R2 = 0.289
R2 ajustado = 0.288
F = 179.397
p global < 0.001
```

Diagnóstico de supuestos:

- Multicolinealidad: casi todos los VIF fueron menores que 5. `Humedad` quedó ligeramente por encima (`VIF = 5.321`) y `Temperatura` cerca del umbral (`VIF = 4.780`), lo cual sugiere colinealidad moderada entre variables meteorológicas.
- Homocedasticidad: Breusch-Pagan rechazó homocedasticidad (`p < 0.001`).
- Normalidad de errores: Shapiro-Wilk rechazó normalidad de residuales (`p < 0.001`).
- Independencia: Durbin-Watson fue 1.213, compatible con autocorrelación positiva en los residuales.

Por estas razones, la lectura de significancia se apoya también en errores estándar robustos HC3. La tabla de coeficientes conserva intervalos de confianza y valores-p clásicos y robustos. Los coeficientes de mayor magnitud fueron:

| Variable | Coeficiente |
|---|---:|
| Duracion | 0.225 |
| Humedad | 0.089 |
| Temperatura | -0.080 |
| Distancia | -0.057 |
| Viento | -0.050 |
| PM2.5 | 0.041 |
| O3 | -0.032 |
| CO | -0.029 |

Interpretación: controlando por el resto de variables, listas más largas tienden a tener mayor Shannon, lo que confirma la importancia de ajustar por esfuerzo de muestreo. Ambientalmente, mayor humedad se asocia con mayor Shannon, mientras que mayor temperatura, viento y ozono se asocian con menor Shannon. La interpretación sigue siendo asociativa, no causal.

Como los predictores fueron estandarizados pero `Shannon_Index` quedó en su escala original, los coeficientes principales se interpretan directamente como cambio esperado en Shannon por un aumento de 1 desviación estándar del predictor. Para facilitar lectura en unidades originales, también se desestandarizaron los coeficientes e intervalos robustos HC3 dividiendo por la desviación estándar usada en cada predictor. Ejemplos:

| Variable | Cambio en Shannon por unidad original | IC robusto HC3 |
|---|---:|---|
| Duracion | 0.00264 por minuto | [0.00232, 0.00296] |
| Humedad | 0.00559 por punto porcentual | [0.00359, 0.00758] |
| Temperatura | -0.01991 por °C | [-0.02729, -0.01253] |
| Distancia | -0.02035 por km | [-0.02836, -0.01234] |
| Viento | -0.05582 por m/s | [-0.07569, -0.03595] |
| PM2.5 | 0.00372 por unidad | [0.00146, 0.00598] |
| O3 | -0.00269 por unidad | [-0.00500, -0.00038] |
| CO | -0.07238 por unidad | [-0.12344, -0.02131] |

Como tratamiento de multicolinealidad se mantuvo el modelo principal completo por coherencia ecológica, pero se corrió un modelo de sensibilidad removiendo `Humedad`, la única variable con `VIF > 5`. En ese modelo todos los VIF bajaron de 5 y el ajuste cambió poco:

```text
Modelo principal: R2 ajustado = 0.288
Sin Humedad:      R2 ajustado = 0.284
```

Esto sugiere que la colinealidad entre humedad y temperatura es moderada y no altera sustancialmente la capacidad explicativa global. Para el informe se puede reportar el modelo completo y mencionar la sensibilidad como respaldo.

Salidas generadas:

```text
Outputs/regression_ols_coefficients.csv
Outputs/regression_ols_coefficients_original_scale.csv
Outputs/regression_vif.csv
Outputs/regression_diagnostics.csv
Outputs/regression_ols_summary.txt
Outputs/regression_sensitivity_no_humidity_coefficients.csv
Outputs/regression_sensitivity_no_humidity_vif.csv
Outputs/regression_sensitivity_no_humidity_diagnostics.csv
Figures/Regression/01_regression_coefficients.png
Figures/Regression/02_residuals_vs_fitted.png
Figures/Regression/03_residuals_qqplot.png
```

## 8. PCA, PPCA y Factor Analysis

Se construyó el Notebook 05:

```text
Notebooks/05_pca_ppca_factor_analysis.ipynb
```

También se dejó una versión reproducible como script:

```text
Scripts/11_dimensionality_reduction.py
```

En esta fase solo entraron contaminantes y clima estandarizados. No se incluyeron variables de esfuerzo de eBird. La matriz ambiental usada fue:

```text
PM2.5, PM10, NO2, CO, O3,
Temperatura, Humedad, Viento, Lluvia, Radiacion_Solar
```

La base usada fue `analysis_env_complete.parquet`, con 6,178 filas completas y 10 variables ambientales.

Resultados principales de PCA:

| Componente | Autovalor | Varianza explicada | Varianza acumulada |
|---|---:|---:|---:|
| PC1 | 4.139 | 41.4% | 41.4% |
| PC2 | 2.486 | 24.9% | 66.2% |
| PC3 | 0.990 | 9.9% | 76.1% |
| PC4 | 0.652 | 6.5% | 82.7% |

El scree plot muestra que los dos primeros componentes concentran cerca de dos tercios de la estructura ambiental. PC1 representa principalmente un gradiente meteorológico y fotoquímico: temperatura, radiación solar, ozono y viento cargan en una dirección, mientras humedad carga en la dirección opuesta. PC2 concentra material particulado, especialmente `PM2.5` y `PM10`.

El biplot analítico proyecta las listas en el espacio PC1-PC2 y colorea los puntos según baja o alta diversidad (`Q1` vs `Q3`). Visualmente, las listas de alta diversidad se desplazan hacia el lado asociado con mayor humedad y menor temperatura/radiación/ozono, consistente con lo observado en EDA, inferencia multivariable y regresión.

Para PPCA se usaron 4 componentes latentes. La varianza de ruido estimada fue:

```text
sigma2 = 0.289
```

Esta cantidad resume la varianza ambiental residual no capturada por los componentes principales bajo el supuesto de ruido isotrópico. En el contexto RMCAB, se interpreta como una medida agregada de variabilidad no explicada por la estructura latente dominante, no como error puro de sensor individual.

En Factor Analysis con rotación Varimax se obtuvieron factores físicamente más interpretables:

- `Factor1`: gradiente meteorológico/luz-humedad, con cargas altas en temperatura, radiación solar, ozono y viento, y carga negativa fuerte en humedad.
- `Factor2`: material particulado, dominado por `PM2.5` y `PM10`.
- `Factor3`: gases de combustión, dominado por `NO2` y `CO`.
- `Factor4`: factor débil, con señal menor en viento y lluvia; debe interpretarse con cautela.

Comparación técnica:

| Método | Qué optimiza | Ventaja principal | Lectura en este proyecto |
|---|---|---|---|
| PCA | Varianza total | Resume la nube ambiental en ejes ortogonales | Útil para biplot y varianza explicada |
| PPCA | Estructura latente + ruido isotrópico | Estima `sigma2` residual | Resume ruido/variabilidad no capturada por 4 componentes |
| FA Varimax | Covarianza compartida | Produce factores más interpretables | Separa meteorología, partículas y gases |

Salidas generadas:

```text
Outputs/dimred_pca_explained_variance.csv
Outputs/dimred_pca_loadings.csv
Outputs/dimred_factor_analysis_varimax_loadings.csv
Outputs/dimred_factor_analysis_uniqueness.csv
Outputs/dimred_method_comparison.csv
Outputs/dimred_scores.csv
Figures/Dimensionality_Reduction/01_pca_scree_plot.png
Figures/Dimensionality_Reduction/02_pca_biplot_q1_q3.png
Figures/Dimensionality_Reduction/03_pca_loadings_heatmap.png
Figures/Dimensionality_Reduction/04_factor_analysis_varimax_loadings.png
```
