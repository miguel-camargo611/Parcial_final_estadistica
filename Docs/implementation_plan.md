# Plan de Implementación — Examen Final
## "Determinantes Ambientales de la Diversidad de Aves a Escala Micro-Hábitat en Bogotá"
### Estadística Multivariada y Modelos Lineales Aplicados

> **Autor:** Miguel Ángel Camargo Mora  
> **Datos:** RMCAB (Red de Monitoreo de Calidad del Aire de Bogotá) · eBird Colombia-DC  
> **Período de análisis:** 2020–2025  
> **Resolución Analítica:** Nivel Lista de eBird (Cruzada por Hora y Estación más cercana)

---

## 1. Arquitectura de Datos: El "Match" Espacio-Temporal

Una fila del dataset final representa un **evento de muestreo de 15-60 min**.

| Bloque | Variables | Función en el Modelo |
|---|---|---|
| **Respuesta Ecológica ($Y$)** | **Índice de Shannon ($H'$)** | Lo que queremos predecir/entender. |
| **Control de Esfuerzo ($C$)** | Duración (min), Distancia (km) | Variables de control (solo van en Regresión). |
| **Predictores Químicos ($X_q$)** | PM2.5, NO₂, O₃, CO | Conforman el "Ambiente Multivariado". |
| **Predictores Físicos ($X_f$)** | Temperatura, Humedad, Viento, Lluvia, Radiación Solar | Conforman el "Ambiente Multivariado". |

---

## 2. Plan de Análisis y Verificación de Supuestos

### 🧹 Fase 1 — Limpieza y Preparación de Datos

- **Identificación de Atípicos (Outliers):** Uso de Z-scores ($Z > 3$) y Rango Intercuartílico (IQR). En datos ambientales, un pico de PM2.5 suele ser real (incendio, inversión térmica). Documentaremos su tratamiento.
- **Valores Nulos:** Imputación lineal para baches de $<3$ horas. Descarte de listas sin match ambiental.
- **Transformación Logarítmica:** Las transformaciones $\ln(x+1)$ se calcularán como diagnóstico auxiliar, pero la matriz principal conservará las variables ambientales en escala original para mantener interpretabilidad física.
- **🚨 ESTANDARIZACIÓN UNIVERSAL (Crucial):** **Absolutamente todas las variables continuas** predictoras (contaminantes, clima y esfuerzo de muestreo) serán estandarizadas con `StandardScaler` (Z-scores) cuando el método lo requiera. Hacerlo para PCA, PPCA, FA e inferencia por distancias garantiza que las métricas no colapsen por diferencias de escala. En regresión permite que los coeficientes ($\beta$) sean directamente comparables entre sí.

### 📊 Fase 2 — Análisis Exploratorio (EDA) (15% de la nota)

- **Descriptivas:** Tabla general (medias, CV, asimetría).
- **Visualización:** Scatter matrix cruzando variables clave. 
- **Heatmap de Correlación:** Matriz de Pearson.

### 📗 Fase 3 — Distribuciones Conjuntas, Marginales y Condicionales (15%)

> *Aclaración sobre dimensionalidad:* Aunque el vector ambiente tiene 10 dimensiones, la visualización de la densidad conjunta es imposible más allá de 3D. El profesor espera que escojas un **par bivariado representativo** (ej. `PM2.5 estandarizado` vs `Temperatura estandarizada`) para la demostración visual, mientras que las fórmulas en Markdown sí se presentan con notación matricial general para $p$ dimensiones.

- **Conjunta (Bivariada):** KDE 2D y contour plot para el par clave seleccionado. 
- **Normal Bivariada:** Ajustar $(\mu, \Sigma)$ y graficar elipses de densidad observada vs. teórica (un gráfico 3D opcional aquí para la "campana" teórica).
- **Condicionales:** $f(H' | \text{Alta Contaminación})$ vs. $f(H' | \text{Baja Contaminación})$.

### 📕 Fase 4 — Inferencia Multivariable (15%)

- **Definición de Grupos:** Usaremos **Cuartiles** del Índice de Shannon:
  - **Alta Diversidad:** Listas en el Cuartil 3 (Q3).
  - **Baja Diversidad:** Listas en el Cuartil 1 (Q1).
- **Prueba de Supuestos Rigurosa:**
  1. *Normalidad Multivariada:* Test de Mardia sobre la matriz ambiental original estandarizada.
  2. *Homocedasticidad Multivariada:* Test M de Box.
- **Inferencia no paramétrica/permutacional:** Comparar la matriz ambiental estandarizada entre Alta y Baja diversidad sin depender de normalidad multivariada estricta. Hotelling T² puede reportarse como referencia clásica, pero la evidencia principal será permutacional.
- **Gráfico:** Elipses de confianza al 95% para ambos centroides bivariados principales.

### 📙 Fase 5 — Regresión Lineal Múltiple (10%)

- **El Modelo Base:** 
  $$Shannon \sim Contaminantes\_Z + Clima\_Z + Duracion\_Z + Distancia\_Z$$
- **Prueba de Supuestos:**
  1. *No-Multicolinealidad:* Variance Inflation Factor ($VIF < 5$).
  2. *Homocedasticidad de errores:* Breusch-Pagan test y gráfico Residuales vs Ajustados.
  3. *Normalidad de errores:* Shapiro-Wilk y QQ-plot.
  4. *Independencia:* Durbin-Watson.
- **Interpretación:** Lectura de los coeficientes estandarizados.

### 📒 Fase 6 — Reducción Dimensional: PCA, PPCA y Factor Analysis (15%)

> *Aclaración:* Solo entran Contaminantes + Clima estandarizados (sin variables de esfuerzo de eBird). 

- **PCA Clásico:** Scree plot, matriz de varianza explicada.
- **Biplot Analítico 🌟:** Graficaremos los *scores* del PCA (los puntos proyectados) en los dos componentes principales, pero **los colorearemos según sean de Alta o Baja Diversidad (Q3 vs Q1)**. Esto nos permitirá ver visualmente si el hábitat de alta diversidad se agrupa en un rincón específico del espacio latente ambiental.
- **PPCA:** Estimar la varianza del ruido ($\sigma^2$) de la red de monitoreo RMCAB.
- **Factor Analysis (FA) con Rotación Varimax:** Rotación de ejes para forzar que cada variable cargue fuerte en un solo factor, facilitando su interpretación física.
- **Comparación:** Tabla técnica PCA vs PPCA vs FA.

---

## 3. Hoja de Ruta de Ejecución

1. **Script de Ingesta:** Extraer RMCAB y eBird.
2. **Script de Geoprocesamiento (El Match):** Unir lat/lon de eBird con estaciones.
3. **Desarrollo de Notebooks:** Fases 1 a 6.
4. **Informe Final:** PDF de consolidación.
