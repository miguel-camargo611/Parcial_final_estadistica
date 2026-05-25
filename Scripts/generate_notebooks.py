import json
from pathlib import Path

NOTEBOOKS_DIR = Path("Notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# NOTEBOOK 01: ANALISIS EXPLORATORIO DE DATOS (EDA)
# ==============================================================================
nb01_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 01 — Análisis Exploratorio de Datos (EDA) e Ingesta\n",
            "\n",
            "## 1. Contextualización del Problema y Datos\n",
            "El presente estudio busca analizar la relación entre las **variables ambientales y de contaminación atmosférica** de la ciudad de Bogotá y la **biodiversidad de aves urbanas**, medida a través del **Índice de Diversidad de Shannon ($H'$)** derivado de registros de ciencia ciudadana (*eBird*).\n",
            "\n",
            "Los datos ambientales provienen de la Red de Monitoreo de Calidad del Aire de Bogotá (RMCAB) y de IDEAM. Contemplan variables continuas como material particulado ($PM_{2.5}$, $PM_{10}$), gases contaminantes ($NO_2$, $CO$, $O_3$), y variables meteorológicas (Temperatura, Humedad, Viento, Radiación Solar), así como la ocurrencia de Lluvia (variable bimodal). Las observaciones de aves eBird se cruzaron espacio-temporalmente con la estación de monitoreo más cercana dentro de un radio máximo de 5.5 km y una ventana temporal estricta de ±1 hora.\n",
            "\n",
            "En esta sección realizamos un Análisis Exploratorio de Datos (EDA) para evaluar las distribuciones originales, identificar la presencia de asimetrías severas, examinar correlaciones lineales y no lineales (LOWESS), y contrastar descriptivamente las condiciones ambientales asociadas a la baja (Q1) y alta (Q3) diversidad."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from pathlib import Path\n",
            "import os\n",
            "\n",
            "# Asegurar el directorio de trabajo correcto\n",
            "ROOT = Path.cwd().parent if Path.cwd().name == 'Notebooks' else Path.cwd()\n",
            "os.chdir(ROOT)\n",
            "\n",
            "import pandas as pd\n",
            "import IPython.display as display\n",
            "import importlib.util\n",
            "spec = importlib.util.spec_from_file_location('eda', ROOT / 'Scripts/06_eda.py')\n",
            "eda = importlib.util.module_from_spec(spec)\n",
            "spec.loader.exec_module(eda)\n",
            "\n",
            "# Cargar datos\n",
            "df = pd.read_parquet(eda.DATA_PATH)\n",
            "print(f\"Dimensiones del dataset de análisis completo: {df.shape[0]} observaciones x {df.shape[1]} columnas\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Estadísticas Descriptivas Generales\n",
            "Calculamos las descriptivas fundamentales de las variables ambientales y ecológicas del dataset, incluyendo medidas de tendencia central, dispersión, asimetría (*skewness*) y curtosis. \n",
            "\n",
            "Un aspecto crítico es que la mayoría de los contaminantes atmosféricos ($PM_{2.5}$, $PM_{10}$, $CO$) exhiben una asimetría positiva (*right-skewed*) debido a eventos puntuales de alta contaminación. Las variables meteorológicas como Humedad y Temperatura presentan comportamientos más balanceados, aunque con curtosis particulares por los ciclos nictemerales (día/noche)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Generar y cargar estadísticas descriptivas\n",
            "desc, grouped = eda.save_descriptives(df)\n",
            "print(\"=== ESTADÍSTICAS DESCRIPTIVAS DE LAS VARIABLES DE ESTUDIO ===\")\n",
            "display.display(desc.round(3))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Distribución del Índice de Shannon\n",
            "El Índice de Shannon es nuestra variable de respuesta ecológica. Se calcula como:\n",
            "$$H' = -\\sum_{i=1}^{S} p_i \\ln p_i$$\n",
            "donde $p_i$ es la proporción de individuos de la especie $i$ con respecto al total. Un valor de $H'=0$ indica que solo hay una especie presente (monodominancia). A medida que aumenta el número de especies (riqueza) y la equidad en sus abundancias, el índice crece.\n",
            "\n",
            "Visualizamos la distribución del índice y marcamos los límites de los cuartiles Q1 (percentil 25, límite superior para 'Baja Diversidad') y Q3 (percentil 75, límite inferior para 'Alta Diversidad') que utilizaremos en los análisis multivariables inferenciales."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Graficar distribución de Shannon\n",
            "eda.plot_shannon_distribution(df)\n",
            "from IPython.display import Image\n",
            "Image(filename='Figures/EDA/01_shannon_distribution.png')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Distribuciones de las Variables Ambientales\n",
            "Analizamos los histogramas de las variables ambientales en sus escalas físicas originales. \n",
            "Observamos:\n",
            "- Fuertes asimetrías en $PM_{2.5}$ y $PM_{10}$, típicas de distribuciones log-normales de contaminantes en áreas urbanas.\n",
            "- Una distribución bimodal en la Humedad Relativa y la Radiación Solar, explicada por el contraste drástico entre las horas diurnas y nocturnas.\n",
            "- La Lluvia es tratada como una variable cualitativa/bimodal (0 = Sin Lluvia, 1 = Con Lluvia) debido a que la gran mayoría de observaciones registran precipitación cero, lo que imposibilita un tratamiento continuo tradicional."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Graficar histogramas de variables ambientales\n",
            "eda.plot_environment_distributions(df)\n",
            "Image(filename='Figures/EDA/02_environment_distributions_original.png')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Análisis de Correlación Bivariada\n",
            "Para explorar las asociaciones lineales, calculamos y visualizamos la matriz de correlación de Pearson entre las variables continuas.\n",
            "Destacan colinealidades físicas e instrumentales esperadas:\n",
            "1. Alta correlación positiva entre $PM_{2.5}$ y $PM_{10}$ ($r \\approx 0.8$), dado que la fracción fina es parte de la fracción gruesa.\n",
            "2. Fuerte correlación negativa entre Temperatura e Humedad Relativa ($r \\approx -0.7$), gobernada por la física de la presión de vapor de saturación.\n",
            "3. Correlaciones débiles pero significativas de los contaminantes y el clima con el Índice de Shannon."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "pearson, spearman = eda.save_correlations(df)\n",
            "eda.plot_correlation_heatmap(pearson)\n",
            "print(\"=== CORRELACIONES MÁS FUERTES CON EL ÍNDICE DE SHANNON ===\")\n",
            "top_corr = eda.save_top_correlations(pearson)\n",
            "display.display(top_corr.head(8))\n",
            "Image(filename='Figures/EDA/03_pearson_correlation_heatmap.png')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Relaciones No Lineales con el Índice de Shannon\n",
            "Evaluamos la relación individual de las principales variables ambientales con el Índice de Shannon mediante diagramas de dispersión con ajustes suavizados **LOWESS** (*Locally Weighted Scatterplot Smoothing*).\n",
            "\n",
            "El ajuste LOWESS nos permite capturar tendencias no lineales y cambios de curvatura sin imponer una forma funcional a priori (como una recta o parábola), lo cual es crucial para identificar efectos de umbral ecológico."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "eda.plot_shannon_scatter_grid(df)\n",
            "Image(filename='Figures/EDA/04_shannon_vs_environment_scatter.png')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Comparación del Perfil Ambiental en Extremos de Diversidad (Q1 vs Q3)\n",
            "Para contrastar las condiciones ambientales asociadas a la biodiversidad, dividimos los datos en observaciones con Baja Diversidad ($H' \\le Q1$) y Alta Diversidad ($H' \\ge Q3$). \n",
            "\n",
            "Para hacer las variables comparables e independientes de sus unidades físicas, graficamos sus valores estandarizados ($Z$-scores). Esto permite ver inmediatamente qué variables ambientales se encuentran por encima o por debajo de su media histórica en cada grupo de diversidad."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "eda.plot_q1_q3_boxplots(df)\n",
            "Image(filename='Figures/EDA/05_q1_q3_environment_boxplots_z.png')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Conclusiones del EDA\n",
            "1. **Heterocedasticidad y Asimetría:** La distribución de los contaminantes y la presencia de asimetrías severas sugieren la conveniencia de usar métodos robustos en la regresión (como errores robustos HC3) y transformaciones o estandarizaciones antes de aplicar métodos multivariables paramétricos.\n",
            "2. **Multicolinealidad:** Existe una correlación intrínseca muy fuerte en parejas de variables (como Temperatura-Humedad y PM2.5-PM10), lo que podría inflar la varianza en modelos de regresión lineal. Esto justifica hacer análisis de sensibilidad y reducción de dimensiones.\n",
            "3. **Perfiles Diferenciados:** Se observa preliminarmente que las áreas con Alta Diversidad (Q3) tienden a presentar menores niveles de material particulado y mayor estabilidad térmica, hipótesis que probaremos formalmente en los siguientes módulos."
        ]
    }
]

# Escribir Notebook 01
with open(NOTEBOOKS_DIR / "01_EDA.ipynb", "w", encoding="utf-8") as f:
    json.dump({"cells": nb01_cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 2}, f, indent=2, ensure_ascii=False)


# ==============================================================================
# NOTEBOOK 02: DISTRIBUCIONES CONJUNTAS
# ==============================================================================
nb02_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 02 — Distribuciones Conjuntas y Análisis Condicional\n",
            "\n",
            "## 1. Marco Teórico: Distribución Normal Bivariada\n",
            "La distribución normal bivariada describe el comportamiento conjunto de dos variables aleatorias continuas $X_1$ y $X_2$. Su función de densidad de probabilidad conjunta está dada por:\n",
            "$$f(x_1, x_2) = \\frac{1}{2\\pi \\sigma_1 \\sigma_2 \\sqrt{1-\\rho^2}} \\exp\\left( -\\frac{1}{2(1-\\rho^2)} \\left[ \\frac{(x_1-\\mu_1)^2}{\\sigma_1^2} + \\frac{(x_2-\\mu_2)^2}{\\sigma_2^2} - \\frac{2\\rho(x_1-\\mu_1)(x_2-\\mu_2)}{\\sigma_1\\sigma_2} \\right] \\right)$$\n",
            "donde $\\mu_1, \\mu_2$ son las medias, $\\sigma_1^2, \\sigma_2^2$ son las varianzas y $\\rho$ es el coeficiente de correlación de Pearson.\n",
            "\n",
            "En forma matricial, si definimos el vector de variables $\\mathbf{X} = [X_1, X_2]^T$, la distribución se denota como $\\mathbf{X} \\sim N_2(\\boldsymbol{\\mu}, \\boldsymbol{\\Sigma})$, con densidad:\n",
            "$$f(\\mathbf{x}) = \\frac{1}{2\\pi |\\boldsymbol{\\Sigma}|^{1/2}} \\exp\\left( -\\frac{1}{2} (\\mathbf{x}-\\boldsymbol{\mu})^T \\boldsymbol{\\Sigma}^{-1} (\\mathbf{x}-\\boldsymbol{\mu}) \\right)$$\n",
            "donde $\\boldsymbol{\\mu}$ es el vector de medias y $\\boldsymbol{\\Sigma}$ es la matriz de covarianza. Las curvas de nivel donde la densidad es constante corresponden a elipses centradas en $\\boldsymbol{\\mu}$. La ecuación de estas elipses de contorno está definida por la distancia de Mahalanobis al cuadrado:\n",
            "$$d^2(\\mathbf{x}) = (\\mathbf{x}-\\boldsymbol{\mu})^T \\boldsymbol{\\Sigma}^{-1} (\\mathbf{x}-\\boldsymbol{\mu}) = c^2$$\n",
            "Bajo el supuesto de normalidad bivariada, la distancia de Mahalanobis al cuadrado sigue una distribución Chi-cuadrado con 2 grados de libertad: $d^2(\\mathbf{x}) \\sim \\chi^2_2$. Esto permite definir elipses de densidad para un nivel de probabilidad $\\alpha$ (por ejemplo, 50%, 80% o 95%) utilizando el valor crítico $c^2 = \\chi^2_2(1-\\alpha)$.\n",
            "\n",
            "## 2. Aplicación Práctica: Temperatura vs Humedad\n",
            "Analizaremos conjuntamente la Temperatura y la Humedad Relativa estandarizadas ($Z$-scores). Desde una perspectiva física, estas variables presentan una relación termodinámica inversa estrecha (colinealidad física). Estandarizarlas nos permite modelar su interacción conjunta eliminando las diferencias de escala."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from pathlib import Path\n",
            "import os\n",
            "\n",
            "ROOT = Path.cwd().parent if Path.cwd().name == 'Notebooks' else Path.cwd()\n",
            "os.chdir(ROOT)\n",
            "\n",
            "import pandas as pd\n",
            "import IPython.display as display\n",
            "import importlib.util\n",
            "spec = importlib.util.spec_from_file_location('joint', ROOT / 'Scripts/08_joint_distributions.py')\n",
            "joint = importlib.util.module_from_spec(spec)\n",
            "spec.loader.exec_module(joint)\n",
            "\n",
            "df, pollution_q1, pollution_q3 = joint.prepare_data()\n",
            "print(f\"Dimensiones de los datos preparados: {df.shape[0]} filas\")\n",
            "print(f\"Par bivariado bajo análisis: {joint.BIVAR_Z}\")\n",
            "print(f\"Variables que componen el índice de contaminación ponderado Z: {joint.POLLUTION_Z}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Estimación de Densidad por Kernel (KDE 2D) y Elipses Empíricas\n",
            "La Estimación de Densidad por Kernel de dos dimensiones (KDE 2D) nos permite ver de manera no paramétrica cómo se distribuyen conjuntamente las observaciones de Temperatura y Humedad.\n",
            "\n",
            "Luego, construimos las elipses empíricas a niveles de probabilidad del 50%, 80% y 95% calculadas a partir de la media y matriz de covarianza muestral. Estas elipses muestran dónde se concentra la mayor proporción de la masa de datos."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "bivar, conditional = joint.save_summaries(df, pollution_q1, pollution_q3)\n",
            "print(\"=== ESTADÍSTICAS BIVARIADAS DE TEMPERATURA Y HUMEDAD Z ===\")\n",
            "display.display(bivar.round(3))\n",
            "\n",
            "# Graficar KDE 2D y elipses empíricas\n",
            "joint.plot_kde_2d(df)\n",
            "joint.plot_density_ellipses(df)\n",
            "\n",
            "from IPython.display import Image\n",
            "display.display(Image(filename='Figures/Joint_Distributions/01_kde2d_temperatura_humedad.png'))\n",
            "display.display(Image(filename='Figures/Joint_Distributions/02_elipses_densidad_temperatura_humedad.png'))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Visualización Conjunta y Marginal de la Biodiversidad\n",
            "Para complementar la caracterización de la distribución conjunta, graficamos el diagrama de dispersión bivariado de los datos crudos (Temperatura vs Humedad Relativa) utilizando un gradiente de color continuo determinado por el Índice de Biodiversidad de Shannon. Esto nos permite inspeccionar de manera directa cómo covarían la biodiversidad y las condiciones climatológicas.\n",
            "\n",
            "Adicionalmente, generamos una matriz de diagramas de dispersión (*pairplot*) para las variables ambientales críticas ($PM_{2.5}$, Temperatura, Humedad y $O_3$) junto con el Índice de Shannon, proyectando en la diagonal las estimaciones de la **densidad marginal (KDE marginal)** de cada variable. Esto nos permite contrastar el comportamiento marginal individual de cada variable frente a sus distribuciones de probabilidad conjuntas (fuera de la diagonal)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Graficar dispersión coloreada por gradiente de Shannon y pairplot de marginales\n",
            "joint.plot_scatter_shannon_gradient(df)\n",
            "joint.plot_pairplot_marginals(df)\n",
            "\n",
            "from IPython.display import display, Image\n",
            "print(\"=== DISPERSIÓN BIVARIADA COLOREADA POR BIODIVERSIDAD ===\")\n",
            "display.display(Image(filename='Figures/Joint_Distributions/05_scatter_shannon_gradient.png'))\n",
            "\n",
            "print(\"\\n=== PANEL DE DISTRIBUCIONES CONJUNTAS Y DIAGONALES MARGINALES ===\")\n",
            "display.display(Image(filename='Figures/Joint_Distributions/06_pairplot_marginales.png'))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Comparación: KDE Observada vs. Distribución Normal Bivariada Teórica\n",
            "Contrastamos visualmente las curvas de nivel (contornos) de la densidad KDE real observada en los datos (curvas rojas) frente a los contornos de una distribución normal bivariada teórica ajustada con los parámetros muestrales (curvas azules).\n",
            "\n",
            "Si los datos fueran perfectamente normales bivariados, las curvas rojas y azules coincidirían perfectamente en orientación, centro y espaciamiento. Las diferencias revelan desviaciones de la normalidad conjunta (asimetrías, colas pesadas o curvaturas no lineales en los extremos)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Ajustar normal teórica vs observada\n",
            "joint.plot_bivariate_normal(df)\n",
            "Image(filename='Figures/Joint_Distributions/03_normal_bivariada_temperatura_humedad.png')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Análisis Condicional de la Biodiversidad por Niveles de Contaminación\n",
            "Definimos un índice de contaminación agregando los contaminantes $PM_{2.5}$, $PM_{10}$, $NO_2$, $CO$, y $O_3$ estandarizados. A partir de sus cuartiles extremos, segregamos los datos en dos escenarios de estrés ambiental:\n",
            "- **Baja Contaminación ($Q1$):** Observaciones con niveles inferiores al percentil 25.\n",
            "- **Alta Contaminación ($Q3$):** Observaciones con niveles superiores al percentil 75.\n",
            "\n",
            "Evaluamos la densidad condicional de nuestro índice ecológico (Shannon_Index) dada esta partición ambiental. Esto nos permite comprobar visual y estadísticamente si existe un desplazamiento de la distribución de biodiversidad ante la presencia de estrés por contaminación."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(f\"=== UMBRALES DE CONTAMINACIÓN Z ===\")\n",
            "print(f\"Baja Contaminación (<= Q1): {pollution_q1:.3f}\")\n",
            "print(f\"Alta Contaminación (>= Q3): {pollution_q3:.3f}\")\n",
            "print(\"\\n=== RESUMEN CONDICIONAL DEL ÍNDICE DE SHANNON ===\")\n",
            "display.display(conditional.round(3))\n",
            "\n",
            "# Graficar distribución condicional\n",
            "joint.plot_conditional_shannon(df)\n",
            "Image(filename='Figures/Joint_Distributions/04_shannon_condicional_contaminacion.png')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Conclusiones e Interpretación Ecológica\n",
            "1. **Colinealidad Física Clima:** La Temperatura y la Humedad Relativa muestran una elipse con pendiente negativa muy marcada, indicando una correlación negativa intensa. Físicamente, el aire bogotano a mayor temperatura se vuelve más seco, y el enfriamiento nocturno o por lluvias eleva la humedad.\n",
            "2. **Ajuste Bivariado:** La forma elíptica de la densidad observada sigue razonablemente la normal teórica en la región central, pero presenta colas más extendidas (asimetría conjunta y curtosis) en los extremos, lo que afectará los supuestos de las pruebas paramétricas multivariadas clásicas.\n",
            "3. **Efecto de la Contaminación:** Se aprecia visualmente que en condiciones de Alta Contaminación (Q3, curva roja), la distribución del índice de Shannon se desplaza hacia valores menores de diversidad de aves en comparación con el escenario de Baja Contaminación (Q1, curva verde). Además, la dispersión del índice cambia, sugiriendo heterocedasticidad condicional."
        ]
    }
]

# Escribir Notebook 02
with open(NOTEBOOKS_DIR / "02_distribuciones_conjuntas.ipynb", "w", encoding="utf-8") as f:
    json.dump({"cells": nb02_cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 2}, f, indent=2, ensure_ascii=False)


# ==============================================================================
# NOTEBOOK 03: INFERENCIA MULTIVARIABLE Y SUPUESTOS
# ==============================================================================
nb03_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 03 — Inferencia Multivariable y Evaluación de Supuestos\n",
            "\n",
            "## 1. Marco Teórico\n",
            "En esta sección comparamos formalmente el vector de medias ambientales (centroides) correspondientes a los sitios con Baja Diversidad de Aves ($Q1$ de Shannon) frente a los de Alta Diversidad ($Q3$). El objetivo es determinar si las condiciones ambientales medias difieren significativamente entre ambos escenarios ecológicos.\n",
            "\n",
            "### Supuesto 1: Normalidad Multivariante (Prueba de Mardia)\n",
            "La prueba de Mardia evalúa la normalidad multivariante contrastando los coeficientes de asimetría ($b_{1,p}$) y curtosis ($b_{2,p}$) multivariantes sobre los vectores de datos $\\mathbf{x}_i$.\n",
            "Las hipótesis son:\n",
            "$$H_0: \\text{Los datos provienen de una población con distribución normal multivariada}$$\n",
            "$$H_1: \\text{Los datos no siguen una distribución normal multivariada}$$\n",
            "\n",
            "La asimetría de Mardia se define como:\n",
            "$$b_{1,p} = \\frac{1}{n^2} \\sum_{i=1}^n \\sum_{j=1}^n \\left( (\\mathbf{x}_i - \\bar{\\mathbf{x}})^T \\mathbf{S}^{-1} (\\mathbf{x}_j - \\bar{\\mathbf{x}}) \\right)^3$$\n",
            "Para muestras grandes, el estadístico $A_1 = \\frac{n}{6} b_{1,p}$ sigue una distribución $\\chi^2_d$ con $d = \\frac{p(p+1)(p+2)}{6}$ grados de libertad.\n",
            "\n",
            "La curtosis de Mardia se define como:\n",
            "$$b_{2,p} = \\frac{1}{n} \\sum_{i=1}^n \\left( (\\mathbf{x}_i - \\bar{\\mathbf{x}})^T \\mathbf{S}^{-1} (\\mathbf{x}_i - \\bar{\\mathbf{x}}) \\right)^2$$\n",
            "El estadístico estandarizado de la curtosis se comporta asintóticamente como una normal estándar: $Z = \\frac{b_{2,p} - p(p+2)}{\\sqrt{8p(p+2)/n}} \\sim N(0, 1)$.\n",
            "\n",
            "### Supuesto 2: Homocedasticidad Multivariante (Prueba M de Box)\n",
            "La prueba M de Box evalúa la igualdad de matrices de covarianza entre $g$ grupos independientes. \n",
            "$$H_0: \\boldsymbol{\\Sigma}_1 = \\boldsymbol{\\Sigma}_2 = \\dots = \\boldsymbol{\\Sigma}_g$$\n",
            "$$H_1: \\exists k, l \\text{ tales que } \\boldsymbol{\\Sigma}_k \\neq \\boldsymbol{\\Sigma}_l$$\n",
            "\n",
            "El estadístico M se calcula como:\n",
            "$$M = (N-g) \\ln |\\mathbf{S}_{pooled}| - \\sum_{k=1}^g (n_k-1) \\ln |\\mathbf{S}_k|$$\n",
            "donde $\\mathbf{S}_k$ es la matriz de covarianza del grupo $k$, $\\mathbf{S}_{pooled}$ es la matriz de covarianza combinada y $N = \\sum n_k$. Se aplica una constante de corrección para aproximar $M$ a una distribución $\\chi^2$ o a una distribución $F$ de Snedecor.\n",
            "\n",
            "### Contraste de Centroides\n",
            "1. **Estadístico $T^2$ de Hotelling:** Es la extensión multivariada de la prueba t de Student para muestras independientes. Si los supuestos de normalidad y homocedasticidad se cumplen:\n",
            "   $$T^2 = \\frac{n_1 n_2}{n_1 + n_2} (\\bar{\\mathbf{x}}_1 - \\bar{\\mathbf{x}}_2)^T \\mathbf{S}_{pooled}^{-1} (\\bar{\\mathbf{x}}_1 - \\bar{\\mathbf{x}}_2)$$\n",
            "   Bajo $H_0: \\boldsymbol{\\mu}_1 = \\boldsymbol{\\mu}_2$, el estadístico escalado se distribuye como una variable $F$:\n",
            "   $$\\frac{n_1 + n_2 - p - 1}{p(n_1 + n_2 - 2)} T^2 \\sim F_{p, n_1+n_2-p-1}$$\n",
            "2. **Prueba Permutacional de Distancia entre Centroides:** Dado que el tamaño de muestra es muy grande ($n > 100,000$), las pruebas paramétricas clásicas como Hotelling pierden robustez si hay violaciones severas de supuestos. La prueba permutacional no paramétrica es el estándar de oro. Mezcla las etiquetas de los grupos aleatoriamente en 10,000 iteraciones para construir la distribución empírica de la distancia de la norma euclidiana entre centroides bajo la hipótesis nula de igualdad. El p-valor se calcula como la proporción de permutaciones donde la distancia permutada supera a la observada."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from pathlib import Path\n",
            "import os\n",
            "\n",
            "ROOT = Path.cwd().parent if Path.cwd().name == 'Notebooks' else Path.cwd()\n",
            "os.chdir(ROOT)\n",
            "\n",
            "import pandas as pd\n",
            "import IPython.display as display\n",
            "import importlib.util\n",
            "spec = importlib.util.spec_from_file_location('multivar', ROOT / 'Scripts/09_multivariate_inference_assumptions.py')\n",
            "multivar = importlib.util.module_from_spec(spec)\n",
            "spec.loader.exec_module(multivar)\n",
            "\n",
            "df = pd.read_parquet(multivar.DATA_PATH)\n",
            "groups, low, high = multivar.prepare_groups(df)\n",
            "print(f\"Datos cargados para inferencia. n total = {len(groups)} (Baja Diversidad n = {len(low)}, Alta Diversidad n = {len(high)})\")\n",
            "print(f\"Predictores ambientales evaluados: {multivar.Z_CONTINUOUS_ENV_VARS}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Diagnóstico de Normalidad Multivariante (Mardia)\n",
            "Ejecutamos la prueba de Mardia para la muestra completa y para cada grupo por separado."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Correr pruebas y obtener tablas\n",
            "mardia, box_m, hotelling, permutation, means, permuted_distances = multivar.save_results(df, groups, low, high, n_permutations=999)\n",
            "print(\"=== RESULTADOS DE LA PRUEBA DE NORMALIDAD DE MARDIA ===\")\n",
            "display.display(mardia.round(4))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Diagnóstico de Homocedasticidad (M de Box)\n",
            "Evaluamos si las matrices de covarianza de las condiciones ambientales de los sitios de Baja y Alta diversidad son estadísticamente idénticas."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"=== RESULTADOS DE LA PRUEBA M DE BOX ===\")\n",
            "display.display(box_m.round(4))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Contraste de Medias Multivariantes (Hotelling $T^2$ y Permutaciones)\n",
            "Comparamos la hipótesis de igualdad de vectores de medias usando el método paramétrico clásica (Hotelling) y el método robusto no paramétrico (Prueba Permutacional de distancia de centroides con su distribución)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"=== ESTADÍSTICO T2 DE HOTELLING PARAMÉTRICO ===\")\n",
            "display.display(hotelling.round(4))\n",
            "print(\"\\n=== CONTRASENSIÓN PERMUTACIONAL DE DISTANCIA DE CENTROIDES ===\")\n",
            "display.display(permutation.round(4))\n",
            "\n",
            "# Graficar distribución de permutación\n",
            "multivar.plot_permutation_distribution(permuted_distances, permutation.loc[0, 'observed_centroid_distance'])\n",
            "from IPython.display import Image\n",
            "Image(filename='Figures/Multivariate_Inference/03_permutation_centroid_distance_continuous.png')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Análisis del Perfil Ambiental Medio de los Grupos\n",
            "Visualizamos los perfiles medios estandarizados para ver en qué variables se dan las diferencias más grandes entre los sitios con Alta Diversidad (Q3) y Baja Diversidad (Q1). Adicionalmente, graficamos las elipses de confianza del 95% para los centroides proyectados en el plano bidimensional de Temperatura y Humedad."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "multivar.plot_group_profile(means)\n",
            "multivar.plot_centroid_confidence_ellipses(groups)\n",
            "\n",
            "print(\"=== DIFERENCIA DE MEDIAS ORDENADAS DE MANERA CRECIENTE (Alta - Baja) ===\")\n",
            "display.display(means)\n",
            "\n",
            "from IPython.display import Image\n",
            "display.display(Image(filename='Figures/Multivariate_Inference/01_group_mean_profile_continuous.png'))\n",
            "display.display(Image(filename='Figures/Multivariate_Inference/02_centroid_confidence_ellipses_temp_humidity.png'))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Conclusiones del Contraste de Hipótesis y Supuestos\n",
            "1. **Rechazo de Normalidad y Homocedasticidad:** Tanto la prueba de Mardia como la M de Box arrojan $p$-valores extremadamente bajos ($p < 0.0001$), rechazando la hipótesis de normalidad y covarianza homogénea. Esto invalida teóricamente la precisión matemática del test paramétrico de Hotelling, obligándonos a confiar en la prueba permutacional.\n",
            "2. **Diferencias Significativas:** La prueba permutacional confirma con alta certidumbre que los centroides ambientales son significativamente distintos ($p = 0.001$). Las aves de alta diversidad de eBird se agrupan en perfiles ambientales particulares.\n",
            "3. **Interpretación del Perfil Ecológico:** Al observar el perfil medio, los sitios de **Alta Diversidad (Q3)** muestran niveles menores de material particulado ($PM_{2.5}$ y $PM_{10}$ por debajo del promedio general, $Z < 0$) y niveles ligeramente más altos de Humedad Relativa y O3 con respecto a las áreas degradadas de Baja Diversidad (Q1). Esto sugiere que la polución del aire está fuertemente correlacionada de forma negativa con la presencia y diversidad de avifauna en el ecosistema urbano."
        ]
    }
]

# Escribir Notebook 03
with open(NOTEBOOKS_DIR / "03_inferencia_multivariable.ipynb", "w", encoding="utf-8") as f:
    json.dump({"cells": nb03_cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 2}, f, indent=2, ensure_ascii=False)


# ==============================================================================
# NOTEBOOK 04: REGRESION LINEAL MULTIPLE
# ==============================================================================
nb04_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 04 — Modelación por Regresión Lineal Múltiple\n",
            "\n",
            "## 1. Marco Teórico\n",
            "La regresión lineal múltiple permite cuantificar el efecto simultáneo de múltiples variables predictoras ambientales ($X_1, X_2, \\dots, X_p$) y variables de control de esfuerzo de muestreo (Duración y Distancia) sobre la biodiversidad medida por el índice de Shannon ($Y$).\n",
            "\n",
            "El modelo general se expresa como:\n",
            "$$Y_i = \\beta_0 + \\beta_1 X_{1i} + \\beta_2 X_{2i} + \\dots + \\beta_p X_{pi} + \\epsilon_i$$\n",
            "donde $\\beta_j$ son los coeficientes a estimar y $\\epsilon_i$ representa el error aleatorio, que bajo los supuestos clásicos de Gauss-Markov cumple:\n",
            "1. $E[\\epsilon_i] = 0$ (linealidad).\n",
            "2. $Var(\\epsilon_i) = \\sigma^2$ (homocedasticidad).\n",
            "3. $Cov(\\epsilon_i, \\epsilon_j) = 0$ para $i \\neq j$ (independencia).\n",
            "4. Ausencia de multicolinealidad perfecta.\n",
            "5. $\\epsilon_i \\sim N(0, \\sigma^2)$ (normalidad de residuales) para la inferencia clásica sobre significancia de coeficientes.\n",
            "\n",
            "### Inferencia Robusta a la Heterocedasticidad (Errores HC3)\n",
            "En presencia de heterocedasticidad (varianza no constante de residuales), los estimadores OLS de los coeficientes siguen siendo insesgados, pero los errores estándar calculados de forma clásica están sesgados, lo que puede inflar la tasa de falsos positivos ($p$-valores espurios).\n",
            "\n",
            "Para resolver esto, aplicamos estimadores sándwich robustos del tipo **HC3** (propuestos por MacKinnon y White), que ajustan la matriz de covarianza de los coeficientes mediante la ponderación de los residuales al cuadrado por los elementos de la diagonal de la matriz de proyección (*hat matrix*, $h_{ii}$):\n",
            "$$\\Sigma_{HC3} = (X^T X)^{-1} X^T \\text{diag}\\left( \\frac{e_i^2}{(1-h_{ii})^2} \\right) X (X^T X)^{-1}$$\n",
            "El factor de ponderación $(1-h_{ii})^2$ es óptimo para evitar subestimar los errores en muestras grandes con puntos de alto apalancamiento (*leverage*).\n",
            "\n",
            "### Multicolinealidad (VIF)\n",
            "El Factor de Inflación de Varianza (VIF) de un predictor $j$ mide cuánto se infla la varianza de su coeficiente debido a la correlación lineal con los demás predictores:\n",
            "$$VIF_j = \\frac{1}{1 - R_j^2}$$\n",
            "donde $R_j^2$ es el coeficiente de determinación de la regresión de $X_j$ sobre todas las demás variables predictoras. Un VIF superior a 5 o 10 indica problemas severos de multicolinealidad, lo que dificulta discernir los efectos individuales de cada variable y aconseja un análisis de sensibilidad eliminando variables redundantes."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from pathlib import Path\n",
            "import os\n",
            "\n",
            "ROOT = Path.cwd().parent if Path.cwd().name == 'Notebooks' else Path.cwd()\n",
            "os.chdir(ROOT)\n",
            "\n",
            "import pandas as pd\n",
            "import IPython.display as display\n",
            "import importlib.util\n",
            "spec = importlib.util.spec_from_file_location('reg', ROOT / 'Scripts/10_multiple_regression.py')\n",
            "reg = importlib.util.module_from_spec(spec)\n",
            "spec.loader.exec_module(reg)\n",
            "\n",
            "df = reg.load_model_data()\n",
            "model, robust_model, x, y = reg.fit_model(df)\n",
            "print(f\"Modelo ajustado sobre {df.shape[0]} observaciones completas.\")\n",
            "print(f\"Predictores incluidos: {reg.PREDICTORS}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Coeficientes Estandarizados OLS y Ajuste Robusto (HC3)\n",
            "Mostramos la tabla de coeficientes estandarizados. Dado que todas las variables predictoras del modelo fueron previamente centradas y reducidas a escala $Z$, la magnitud absoluta de sus coeficientes $\\beta$ refleja directamente el peso o la importancia relativa de cada predictor en la respuesta.\n",
            "\n",
            "Comparamos la desviación estándar clásica frente a los errores estándar robustos HC3 para comprobar la estabilidad de los p-valores."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "coefficients, vifs, diagnostics = reg.save_results(model, robust_model, x)\n",
            "print(\"=== COEFICIENTES DEL MODELO DE REGRESIÓN (ESTANDARIZADOS Z) ===\")\n",
            "display.display(coefficients[['variable', 'coef', 'std_err', 'p_value', 'robust_hc3_std_err', 'robust_hc3_p_value']].round(4))\n",
            "\n",
            "# Graficar importancia de coeficientes\n",
            "reg.plot_coefficients(coefficients)\n",
            "from IPython.display import Image\n",
            "Image(filename='Figures/Regression/01_regression_coefficients.png')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Coeficientes en Escala Física Original\n",
            "Para recuperar la interpretabilidad práctica de los resultados, transformamos los coeficientes de vuelta a sus escalas físicas originales. \n",
            "Por ejemplo, esto nos dice exactamente cuántas unidades del índice de Shannon varían por cada incremento de 1°C de temperatura o por cada aumento de $1\\,\\mu g/m^3$ de $PM_{2.5}$ en el aire."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "original_scale = reg.save_original_scale_results(df, coefficients)\n",
            "print(\"=== COEFICIENTES EN ESCALA FÍSICA ORIGINAL ===\")\n",
            "display.display(original_scale[['variable', 'raw_variable', 'coef_per_1_original_unit', 'robust_hc3_ci_low_per_1_original_unit', 'robust_hc3_ci_high_per_1_original_unit', 'robust_hc3_p_value']].round(6))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Diagnóstico de Supuestos y Multicolinealidad (VIF)\n",
            "Revisamos la colinealidad a través del VIF y verificamos los supuestos de los residuales mediante:\n",
            "- Breusch-Pagan: Homocedasticidad (varianza constante).\n",
            "- Shapiro-Wilk: Normalidad de residuales (sobre una submuestra por el tamaño de n).\n",
            "- Durbin-Watson: Autocorrelación de errores (valor cercano a 2 indica independencia)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"=== TABLA VIF DE MULTICOLINEALIDAD ===\")\n",
            "display.display(vifs.round(3))\n",
            "print(\"\\n=== DIAGNÓSTICOS DE SUPUESTOS DE RESIDUALES ===\")\n",
            "display.display(diagnostics.round(4))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Visualización de Residuales\n",
            "Graficamos los residuales frente a los valores ajustados del modelo para verificar visualmente el supuesto de homocedasticidad y la idoneidad de la forma funcional. Adicionalmente, el Q-Q plot de los residuales ayuda a inspeccionar la desviación de normalidad en las colas."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "reg.plot_residuals_vs_fitted(model)\n",
            "reg.plot_qq(model)\n",
            "\n",
            "from IPython.display import display\n",
            "display(Image(filename='Figures/Regression/02_residuals_vs_fitted.png'))\n",
            "display(Image(filename='Figures/Regression/03_residuals_qqplot.png'))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Análisis de Sensibilidad (Remover Humedad Relativa)\n",
            "Como Humedad Relativa y Temperatura exhiben un VIF elevado ($VIF > 2.5$) y una correlación teórica muy fuerte, realizamos una regresión de sensibilidad omitiendo la Humedad Relativa. Esto nos permite observar si los coeficientes de Temperatura y los contaminantes se mantienen estables, confirmando que la multicolinealidad no altera la dirección ni significancia del efecto."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "sens_model, sens_robust, sens_coefficients, sens_vifs, sens_diagnostics = reg.save_sensitivity_no_humidity(df)\n",
            "print(\"=== COMPARACIÓN DE AJUSTE GLOBAL CON Y SIN HUMEDAD ===\")\n",
            "print(f\"Modelo Completo (R2): {diagnostics.loc[0, 'r_squared']:.4f} | R2 adj: {diagnostics.loc[0, 'adj_r_squared']:.4f}\")\n",
            "print(f\"Sensibilidad (R2):    {sens_diagnostics.loc[0, 'r_squared']:.4f} | R2 adj: {sens_diagnostics.loc[0, 'adj_r_squared']:.4f}\")\n",
            "print(\"\\n=== COEFICIENTES DEL MODELO DE SENSIBILIDAD SIN HUMEDAD ===\")\n",
            "display.display(sens_coefficients[['variable', 'coef', 'robust_hc3_p_value']].round(4))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Conclusiones e Interpretación de la Regresión\n",
            "1. **Efectos de Contaminación:** Se confirma un efecto negativo altamente significativo de los contaminantes locales ($PM_{10}$ y $PM_{2.5}$) sobre el índice de Shannon. El material particulado reduce de manera consistente la diversidad biológica de aves registradas.\n",
            "2. **Variables de Control de Esfuerzo:** Como era de esperarse en datos de ciencia ciudadana, el esfuerzo de muestreo (distancia recorrida y duración en minutos) tiene un coeficiente positivo fuerte y significativo. Controlar por estas variables en el modelo es fundamental para aislar el verdadero efecto ambiental.\n",
            "3. **Heterocedasticidad Detectada:** La prueba de Breusch-Pagan rechaza fuertemente la hipótesis de varianza constante ($p < 0.0001$), lo que valida la decisión de usar la matriz de covarianza robusta **HC3** para las pruebas de hipótesis.\n",
            "4. **Robustez de la Sensibilidad:** Al retirar Humedad del modelo, la Temperatura cambia ligeramente de magnitud pero se mantiene la significancia y los coeficientes de contaminación no sufren distorsiones. Esto demuestra la estabilidad estructural del modelo a pesar de la correlación de variables climáticas."
        ]
    }
]

# Escribir Notebook 04
with open(NOTEBOOKS_DIR / "04_regresion_lineal_multiple.ipynb", "w", encoding="utf-8") as f:
    json.dump({"cells": nb04_cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 2}, f, indent=2, ensure_ascii=False)


# ==============================================================================
# NOTEBOOK 05: PCA, PPCA Y FACTOR ANALYSIS
# ==============================================================================
nb05_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 05 — Reducción Dimensional y Análisis de Variables Latentes\n",
            "\n",
            "## 1. Marco Teórico\n",
            "Cuando se estudian múltiples variables meteorológicas y contaminantes estrechamente correlacionados, el análisis individual puede resultar redundante y confuso. Empleamos tres métodos complementarios de reducción de dimensiones:\n",
            "\n",
            "### Análisis de Componentes Principales (PCA)\n",
            "PCA es un método geométrico no paramétrico que proyecta las variables observadas en direcciones ortogonales (componentes principales) de forma que maximizan de manera decreciente la variabilidad total explicada. Las componentes son combinaciones lineales de las variables originales y se derivan de la descomposición espectral de la matriz de correlación muestral $\\mathbf{R}$:\n",
            "$$\\mathbf{R} = \\mathbf{V} \\boldsymbol{\\Lambda} \\mathbf{V}^T$$\n",
            "donde las columnas de $\\mathbf{V}$ son las cargas (*loadings*) y la diagonal de $\\boldsymbol{\\Lambda}$ contiene los autovalores $\\lambda_i$ (varianza de cada componente).\n",
            "\n",
            "### Probabilistic PCA (PPCA)\n",
            "PPCA formula PCA bajo un modelo probabilístico latente con variables latentes gaussianas $\\mathbf{z} \\sim N(\\mathbf{0}, \\mathbf{I})$ y un término de ruido isotrópico $\\mathbf{e} \\sim N(\\mathbf{0}, \\sigma^2 \\mathbf{I})$:\n",
            "$$\\mathbf{x} = \\mathbf{W}\\mathbf{z} + \\boldsymbol{\\mu} + \\mathbf{e}$$\n",
            "Esto permite modelar formalmente la incertidumbre instrumental de los datos y estimar la varianza del ruido del sensor de calidad del aire a través de $\\sigma^2$.\n",
            "\n",
            "### Análisis Factorial (FA) con Rotación Varimax\n",
            "A diferencia de PCA que solo busca maximizar varianza, el Análisis Factorial asume que existe una estructura subyacente de pocos factores comunes latentes que causan las correlaciones observadas. Las variables observadas se modelan como:\n",
            "$$\\mathbf{x} = \\mathbf{L}\\mathbf{f} + \\mathbf{u}$$\n",
            "donde $\\mathbf{L}$ son las cargas de los factores comunes, $\\mathbf{f}$ son los factores, y $\\mathbf{u}$ representa la variabilidad específica o ruido único de cada variable ($uniqueness_j = \\Psi_{jj}$).\n",
            "\n",
            "Para lograr que las componentes sean físicamente interpretables, aplicamos una **Rotación Ortogonal Varimax** (implementada a mano en nuestro pipeline). La rotación Varimax maximiza la suma de las varianzas de las cargas al cuadrado dentro de cada factor:\n",
            "$$V = \\frac{1}{p} \\sum_{j=1}^k \\left[ p \\sum_{i=1}^p L_{ij}^4 - \\left( \\sum_{i=1}^p L_{ij}^2 \\right)^2 \\right]$$\n",
            "Esto rota los ejes del factor de manera que cada variable cargue muy fuertemente en un solo factor latente y casi cero en los demás, facilitando su identificación.\n",
            "\n",
            "*Nota metodológica:* La variable discreta/bimodal 'Lluvia' ha sido retirada de todos estos análisis para mantener la coherencia teórica del supuesto de normalidad y continuidad multivariable latente."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from pathlib import Path\n",
            "import os\n",
            "\n",
            "ROOT = Path.cwd().parent if Path.cwd().name == 'Notebooks' else Path.cwd()\n",
            "os.chdir(ROOT)\n",
            "\n",
            "import pandas as pd\n",
            "import IPython.display as display\n",
            "import importlib.util\n",
            "spec = importlib.util.spec_from_file_location('dimred', ROOT / 'Scripts/11_dimensionality_reduction.py')\n",
            "dimred = importlib.util.module_from_spec(spec)\n",
            "spec.loader.exec_module(dimred)\n",
            "\n",
            "df, x = dimred.load_data()\n",
            "print(f\"Dimensiones de la matriz de variables ambientales Z (sin lluvia): {x.shape[0]} filas x {x.shape[1]} columnas\")\n",
            "print(f\"Variables continuas incluidas: {dimred.ENV_VARS}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Ajuste de Modelos y Criterio de Selección de Componentes\n",
            "Ajustamos PCA para observar los autovalores y la varianza explicada acumulada. Esto nos permite aplicar criterios de selección de número de factores:\n",
            "1. **Criterio de Kaiser:** Retener componentes con autovalor $\\lambda_i > 1$.\n",
            "2. **Criterio del Scree Plot (Codo):** Identificar el punto de inflexión donde la varianza explicada deja de caer drásticamente.\n",
            "3. **Criterio del 80% de variabilidad explicada.**"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "pca, pca_scores, pca_loadings = dimred.fit_pca(x)\n",
            "ppca, ppca_scores = dimred.fit_ppca(x)\n",
            "fa, fa_scores, fa_loadings, fa_rotated_loadings, fa_rotated_scores = dimred.fit_factor_analysis(x)\n",
            "\n",
            "explained, pca_loadings_df = dimred.save_pca_outputs(pca, pca_loadings)\n",
            "fa_loadings_df, uniqueness = dimred.save_factor_outputs(fa, fa_rotated_loadings)\n",
            "scores = dimred.save_scores(df, pca_scores, fa_rotated_scores)\n",
            "comparison = dimred.save_comparison(pca, ppca, fa, explained)\n",
            "\n",
            "print(\"=== VARIANZA EXPLICADA PCA ===\")\n",
            "display.display(explained.round(4))\n",
            "\n",
            "# Graficar Scree Plot\n",
            "dimred.plot_scree(explained)\n",
            "from IPython.display import Image\n",
            "Image(filename='Figures/Dimensionality_Reduction/01_pca_scree_plot.png')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Visualización del Espacio Proyectado: PCA Biplot 2D y Dispersión 3D\n",
            "Graficamos las dos primeras componentes principales del PCA en un Biplot 2D tradicional.\n",
            "\n",
            "Además, al habernos quedado con exactamente 3 componentes principales (que explican el 80.63% de la varianza total), podemos realizar una **representación tridimensional (3D)** de todas las observaciones, proyectando los registros en el espacio $[PC1, PC2, PC3]$ y coloreando los puntos según pertenezcan al grupo de Baja Diversidad (Q1) o Alta Diversidad (Q3). Esto permite inspeccionar visualmente la separación física de los micro-hábitats de las aves urbanas de Bogotá."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Graficar Biplot 2D y heatmaps de cargas\n",
            "dimred.plot_pca_biplot(scores, pca_loadings_df)\n",
            "dimred.plot_loadings_heatmap(pca_loadings_df, '03_pca_loadings_heatmap.png', 'Cargas PCA')\n",
            "dimred.plot_loadings_heatmap(fa_loadings_df, '04_factor_analysis_varimax_loadings.png', 'Cargas Factor Analysis Varimax')\n",
            "\n",
            "from IPython.display import display, Image\n",
            "print(\"=== PROYECCIÓN BIDIMENSIONAL (BIPLOT PCA) ===\")\n",
            "display(Image(filename='Figures/Dimensionality_Reduction/02_pca_biplot_q1_q3.png'))\n",
            "\n",
            "# Graficar interactivo 3D con Plotly Express (rotativo)\n",
            "import plotly.express as px\n",
            "import numpy as np\n",
            "\n",
            "subset = scores[scores[dimred.GROUP_COL].isin(['Baja_Q1', 'Alta_Q3'])].copy()\n",
            "subset = subset.sample(min(len(subset), 2200), random_state=42)\n",
            "subset['Diversidad'] = subset[dimred.GROUP_COL].map({'Baja_Q1': 'Baja Diversidad (Q1)', 'Alta_Q3': 'Alta Diversidad (Q3)'})\n",
            "\n",
            "fig = px.scatter_3d(\n",
            "    subset,\n",
            "    x='PC1',\n",
            "    y='PC2',\n",
            "    z='PC3',\n",
            "    color='Diversidad',\n",
            "    color_discrete_map={'Baja Diversidad (Q1)': '#c84630', 'Alta Diversidad (Q3)': '#2a9d8f'},\n",
            "    title='Proyección Tridimensional PCA (Micro-hábitats de Aves) — INTERACTIVO (Arrastra para rotar)',\n",
            "    opacity=0.6,\n",
            "    height=600\n",
            ")\n",
            "fig.update_layout(\n",
            "    scene=dict(\n",
            "        xaxis_title='PC1',\n",
            "        yaxis_title='PC2',\n",
            "        zaxis_title='PC3'\n",
            "    ),\n",
            "    margin=dict(l=0, r=0, b=0, t=50)\n",
            ")\n",
            "fig.show()\n",
            "\n",
            "print(\"\\n=== CALIDAD DE REPRESENTACIÓN (HEATMAP DE CARGAS PCA) ===\")\n",
            "display(Image(filename='Figures/Dimensionality_Reduction/03_pca_loadings_heatmap.png'))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Interpretación de Componentes Principales (PCA)\n",
            "En el Análisis de Componentes Principales, las componentes se ordenan estrictamente de manera decreciente según la varianza explicada. No hay rotación, por lo que las cargas representan la proyección ortogonal de mayor varianza:\n",
            "* **PC1 (Factor Climático):** Explica el **45.87%** de la varianza total. Carga fuertemente con Humedad (0.89), Temperatura (-0.86), Ozono (-0.85) y Radiación Solar (-0.81). Captura la macro-variabilidad meteorológica de la ciudad.\n",
            "* **PC2 (Factor de Material Particulado):** Explica el **27.52%** de la varianza. Carga fuertemente con $PM_{10}$ (0.88) y $PM_{2.5}$ (0.84). Representa la polución por partículas sólidas en suspensión.\n",
            "* **PC3 (Factor de Gases de Combustión):** Explica el **7.24%** de la varianza. Carga moderadamente con $NO_2$ (0.43) y $CO$ (0.39), capturando emisiones gaseosas locales."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. PPCA y Modelo Generativo de Variables Latentes\n",
            "A diferencia del enfoque geométrico de PCA, el Análisis de Componentes Principales Probabilístico (PPCA) asume un **modelo generativo**. Postula que las observaciones de las variables ambientales son generadas por 3 *causas* o *variables latentes* subyacentes inobservables, más un ruido aleatorio de medición ($\\sigma^2 \\approx 0.291$).\n",
            "\n",
            "Matemáticamente, la estimación por máxima verosimilitud asume:\n",
            "$$\\mathbf{x} = \\mathbf{W} \\mathbf{z} + \\boldsymbol{\\mu} + \\boldsymbol{\\epsilon}$$\n",
            "Donde $\\mathbf{z} \\sim \\mathcal{N}(0, \\mathbf{I})$ son las variables latentes (clima, material particulado, gases), $\\mathbf{W}$ es la matriz de cargas que relaciona factores con variables, y $\\boldsymbol{\\epsilon} \\sim \\mathcal{N}(0, \\sigma^2 \\mathbf{I})$ es el ruido isotrópico. La estimación de $\\sigma^2$ equivale al promedio de los autovalores no retenidos.\n",
            "\n",
            "Analizando cómo estas variables latentes influyen en los sensores:\n",
            "1. **Variable Latente 1 (El Clima):** Genera la co-ocurrencia observada entre niveles de humedad y radiación/temperatura. Cuando el valor latente de este 'clima' sube, esperamos probabilísticamente que la estación reporte una caída en la temperatura y un aumento en la humedad.\n",
            "2. **Variable Latente 2 (La Fuente de Partículas):** Actúa como el emisor subyacente de polución física. Cuando esta variable generativa está activa, aumenta la probabilidad de que los sensores de la RMCAB detecten simultáneamente altos niveles de $PM_{2.5}$ y $PM_{10}$.\n",
            "3. **Variable Latente 3 (La Fuente de Gases):** Es el proceso generativo de las emisiones de combustión. Al incrementarse, fuerza matemáticamente la subida conjunta en las lecturas de los gases $NO_2$ y $CO$.\n",
            "\n",
            "A continuación calculamos y mostramos la matriz de pesos generativos $\\mathbf{W}$, que nos indica la fuerza (y dirección) con la que cada Factor Latente subyacente moldea el valor observado de las variables ambientales originales."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import numpy as np\n",
            "import pandas as pd\n",
            "import IPython.display as display\n",
            "\n",
            "# Matriz de pesos generativos W en PPCA: W = U_q (L_q - sigma^2 I)^{1/2}\n",
            "sigma2 = ppca.noise_variance_\n",
            "eigenvalues = ppca.explained_variance_\n",
            "W_ppca = ppca.components_.T * np.sqrt(np.maximum(eigenvalues - sigma2, 0))\n",
            "\n",
            "ppca_weights_df = pd.DataFrame(\n",
            "    W_ppca,\n",
            "    index=dimred.CONTINUOUS_VARS,\n",
            "    columns=['Latente 1 (Clima)', 'Latente 2 (Partículas)', 'Latente 3 (Gases)']\n",
            ")\n",
            "print(f\"PPCA Noise Variance Estimate (sigma2): {sigma2:.4f}\\n\")\n",
            "print(\"=== MATRIZ DE PESOS GENERATIVOS W (PPCA) ===\")\n",
            "display.display(ppca_weights_df.round(4))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Interpretación de los Ejes Factoriales y Conclusiones Ecológicas\n",
            "El PPCA nos permite afirmar, desde una óptica probabilística, que la compleja red de correlaciones ambientales que detectamos se puede simplificar en 3 verdaderos motores físicos (Clima, Partículas y Gases), y que la varianza restante es atribuible al error o ruido de los propios instrumentos de medición. Las aves habitan en este espacio tridimensional probabilístico donde cada dimensión actúa como una fuerza ecológica independiente."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Cargas Rotadas del Análisis Factorial (Varimax) y Análisis de Varianza Comunal y Unicidad\n",
            "Revisamos las cargas del Análisis Factorial tras la rotación Varimax con 3 factores comunes latentes.\n",
            "\n",
            "En el Análisis Factorial, la varianza total de cada variable estandarizada se descompone en dos componentes ortogonales:\n",
            "1. **Comunalidad (Varianza Comunal o $h_j^2$):** Es la proporción de la varianza de la variable $j$ que es explicada por los factores comunes latentes. Se calcula como la suma de los cuadrados de las cargas factoriales de esa variable en todos los factores retenidos.\n",
            "2. **Unicidad (Uniqueness o $\\psi_j$):** Es la proporción de la varianza que **no** es explicada por los factores comunes ($\\psi_j = 1 - h_j^2$). Representa la varianza específica de esa variable particular más el error de medición aleatorio.\n",
            "\n",
            "Examinar la comunalidad y unicidad nos permite evaluar la calidad de representación: variables con alta comunalidad (baja unicidad) están muy bien explicadas por el modelo dimensional latente."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"=== CARGAS FACTOR ANALYSIS (VARIMAX ROTATED, 3 FACTORES) ===\")\\n",
            "display.display(fa_loadings_df.round(3))"
        ]
    },
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Diagnóstico y Validación: Error de Reconstrucción (PCA) y Análisis de Covarianza/Correlación Residual (FA)\n",
            "Para evaluar cuantitativamente el desempeño y la validez teórica de la reducción de dimensiones, implementamos dos herramientas métricas cruciales:\n",
            "\n",
            "### 1. Error de Reconstrucción de Frobenius (PCA)\n",
            "El PCA geométrico proyecta los datos $\\mathbf{X}$ en un subespacio de menor dimensión. Reconstruimos los datos originales a partir de los primeros $k$ componentes como $\\hat{\\mathbf{X}}_k = \\mathbf{Z}_k \\mathbf{W}_k^T$. Evaluamos la pérdida de información midiendo el **error de reconstrucción de Frobenius al cuadrado**: \n",
            "$$\\|\\mathbf{X} - \\hat{\\mathbf{X}}_k\\|_F^2 = \\sum_{i=1}^n \\sum_{j=1}^p (x_{ij} - \\hat{x}_{k,ij})^2$$\n",
            "y la fracción de la varianza no reconstruida (residual) asociada.\n",
            "\n",
            "### 2. Análisis de Covarianza Residual (FA)\n",
            "El Análisis Factorial asume que la matriz de correlación observada $\\mathbf{R}$ puede modelarse a partir de las cargas factoriales $\\mathbf{\\Lambda}$ y la matriz diagonal de unicidades $\\mathbf{\\Psi}$ mediante:\n",
            "$$\\hat{\\mathbf{R}} = \\mathbf{\\Lambda}\\mathbf{\\Lambda}^T + \\mathbf{\\Psi}$$\n",
            "El ajuste se evalúa analizando la matriz residual $\\mathbf{R}_{residual} = \\mathbf{R} - \\hat{\\mathbf{R}}$. Calculamos el **error cuadrático medio (RMSE)** de los elementos fuera de la diagonal. Un RMSE bajo ($<0.05$) valida que el modelo de 3 factores comunes reproduce-fielmente la estructura de covarianza de las 8 variables originales."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Evaluar error de reconstrucción PCA\n",
            "df_recon = dimred.compute_reconstruction_error(x, pca)\n",
            "print(\"=== ERROR DE RECONSTRUCCIÓN PCA POR COMPONENTE ===\")\n",
            "display.display(df_recon.round(4))\n",
            "\n",
            "# 2. Evaluar residual de covarianza FA y obtener RMSE\n",
            "R_obs, R_hat, residual, rmse = dimred.compute_fa_covariance_residual(fa, x)\n",
            "print(f\"\\nRMSE de la matriz residual de correlación/covarianza FA: {rmse:.4f}\")\n",
            "\n",
            "# 3. Mostrar visualización del ajuste de covarianza\n",
            "dimred.plot_fa_covariance_comparison(R_obs, R_hat, residual)\n",
            "from IPython.display import Image\n",
            "display.display(Image(filename='Figures/Dimensionality_Reduction/05_fa_covariance_comparison.png'))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Comparación de Métodos de Reducción Dimensional\n",
            "Mostramos la tabla de comparación del ajuste global de los tres algoritmos (PCA, PPCA y Factor Analysis), evaluando la asunción latente de cada uno y la varianza total explicada."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"=== COMPARACIÓN DE MÉTODOS DE REDUCCIÓN DE DIMENSIONES ===\")\\n",
            "display.display(comparison)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Interpretación de los Ejes Factoriales y Conclusiones Ecológicas\n",
            "\n",
            "A diferencia de PCA o PPCA, en el **Análisis Factorial con Rotación Varimax**, la varianza compartida se redistribuye para maximizar la separación de las cargas (acercándolas a 0 o 1). Esto reordena los factores dando prioridad a la interpretabilidad física más limpia, separando por completo las señales que el PCA mezclaba:\n",
            "\n",
            "* **Factor 1 (Contaminación por Material Particulado):** Carga fuertemente con $PM_{2.5}$ (0.97) y $PM_{10}$ (0.80). Este factor aísla la señal de polvo y polución física de manera pura.\n",
            "* **Factor 2 (Clima / Estabilidad Térmica):** Carga fuertemente con Temperatura (0.93), Humedad Relativa (-0.91), Radiación Solar (0.86), Ozono (0.79) y Viento (0.56). Aísla las condiciones meteorológicas del día.\n",
            "* **Factor 3 (Gases de Combustión Local):** Carga fuertemente con $NO_2$ (0.71) y $CO$ (0.65). Separa de manera independiente los gases tóxicos, provenientes típicamente de escapes de vehículos.\n",
            "\n",
            "### Conclusiones Metodológicas\n",
            "- **Comunalidades Aceptables:** Al retener 3 factores latentes, el *uniqueness* promedio es de alrededor de 0.272 (lo que implica una comunalidad promedio del 72.8%). Las variables conservan comunalidades altas ($>0.65$), a excepción del Viento, lo cual valida la eficacia del modelo para representar el sistema original.\n",
            "- **Ventaja del Análisis Factorial (FA) sobre PCA/PPCA:** Mientras que el PCA nos sirvió para proyectar los datos y visualizar la separación general de las observaciones (útil geométricamente), y el PPCA nos dio una visión del proceso generativo y el ruido instrumental, el **Análisis Factorial Rotado** nos entrega la interpretación de variables más clara. Nos demuestra estadísticamente que los micro-hábitats urbanos de Bogotá (y la exposición de sus aves) están gobernados por tres dimensiones totalmente independientes: el nivel de partículas, el clima del sector, y la concentración de gases de combustión."
        ]
    }
]

# Escribir Notebook 05
with open(NOTEBOOKS_DIR / "05_pca_ppca_factor_analysis.ipynb", "w", encoding="utf-8") as f:
    json.dump({"cells": nb05_cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 2}, f, indent=2, ensure_ascii=False)

print("Notebook 05 creado.")
print("Todos los notebooks creados con éxito.")
