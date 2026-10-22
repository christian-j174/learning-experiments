# Guía de estudio: NOAA AI Learning Journeys

Esta carpeta es una copia de la *Learning Journey Library* del Centro de Inteligencia Artificial de NOAA (NCAI). Son **6 proyectos y 18 notebooks de Jupyter**, cada uno con datos reales de NOAA y un problema de ciencia de la Tierra resuelto con estadística o machine learning (ML).

La guía tiene tres partes:

1. [Mapa del repositorio](#1-mapa-del-repositorio): qué hay y cómo se relaciona.
2. [Plan de práctica](#2-plan-de-práctica): en qué orden estudiarlo y qué hacer en cada sesión.
3. [Cada proyecto explicado](#3-cada-proyecto-explicado): la ciencia y el código, a nivel general.
4. [Cómo correrlo en tu Windows](#4-cómo-correrlo-en-tu-windows): entorno, comandos y problemas conocidos.

El repositorio no es grande (31 MB), así que la guía cubre todos los proyectos.

---

## 1. Mapa del repositorio

| Carpeta | Tema científico | Técnica principal | Notebooks | Dificultad |
|---|---|---|---|---|
| `tcprimed/` | Huracanes vistos por satélite | Lectura de NetCDF, compuestos, red neuronal, regridding | 8 (cap. 0 a 4) | Básica → intermedia |
| `rip_current/` | Corrientes de resaca en playas | Clasificación: regresión logística y red neuronal | 1 | Básica → intermedia |
| `fire_radiative_power/` | Incendios forestales | Random Forest (regresión) | 1 | Intermedia |
| `offshore_wind_profiles/` | Viento en altura sobre el mar | Random Forest + validación cruzada | 1 | Intermedia |
| `echofish/` | Peces y plancton con sonar | Datos en la nube (AWS S3), Zarr, GeoPandas | 3 | Intermedia |
| `geomag-magnet/` | Tormentas geomagnéticas (clima espacial) | LSTM (series de tiempo) + IA explicable | 2 | Avanzada |

Lo que todos tienen en común:

- **Datos abiertos de NOAA**, casi siempre descargados al vuelo desde AWS S3 o Zenodo. Necesitas internet.
- **El mismo esqueleto de notebook**: Overview → Prerequisites → Learning Outcomes → Background (la ciencia) → Software → el tutorial → Exercises → References.
- **Una celda al principio que instala paquetes si faltan** (`subprocess.check_call([sys.executable, '-m', 'pip', 'install', ...])`). Con el entorno de esta guía ya está todo instalado, así que esas celdas no hacen nada.

---

## 2. Plan de práctica

La idea es ir de **leer datos** → **ML clásico** → **redes neuronales** → **series de tiempo y explicabilidad**. Cada bloque se apoya en el anterior. Calcula unas 2 o 3 sesiones de 1 a 2 horas por semana; son unas 8 semanas a ritmo tranquilo.

| Semana | Notebooks | Qué aprendes | Práctica sugerida |
|---|---|---|---|
| 1 | `tcprimed` cap. 0, 1a, 1b, 1c | Qué es un archivo NetCDF, grupos, variables, atributos; graficar datos de satélite | Abre otro archivo de otro huracán del bucket y repite las gráficas |
| 2 | `rip_current` | Clasificación binaria, matriz de confusión, curva ROC, umbral de decisión | Cambia las variables de entrada (`X = ripdata[[...]]`) y compara el AUC |
| 3 | `fire_radiative_power` (parte 0 y 2) | Random Forest, train/test split, error MAE/MSE, importancia de variables | Quita `24hr_roll_frp` de las entradas: ¿cuánto empeora? |
| 4 | `offshore_wind_profiles` | Hiperparámetros, validación cruzada por ubicación, aplicar el modelo a datos nuevos | Prueba otros valores de `n_estimators` y `min_samples_leaf` |
| 5 | `tcprimed` cap. 2 y 3 | Análisis compuesto; tu primera red neuronal de regresión, normalización, overfitting | Agrega más frecuencias de microondas como entrada |
| 6 | `tcprimed` cap. 4 + `echofish` cap. 1 | Proyecciones de mapa y regridding; datos crudos de sonar en S3 | Grafica otra frecuencia del sonar |
| 7 | `echofish` cap. 2 y 3 | Zarr en la nube, diferencia de frecuencias, GeoJSON y mapas interactivos | Cambia el rango de tiempo/profundidad del ecograma |
| 8 | `geomag-magnet` cap. 1 y 2 | LSTM para series de tiempo, guardar/cargar modelos, importancia por permutación | Entrena con la configuración "moderada" y compárala con la rápida |

**Método para cada notebook** (te sirve también para mejorar en resolución de problemas):

1. **Lee primero el Background** sin correr nada. Escribe en una línea: *¿qué se quiere predecir y con qué datos?*
2. **Corre celda por celda** (`Shift+Enter`). Antes de cada celda, predice qué va a salir. Si te equivocas, ahí hay algo que aprender.
3. **Inspecciona los datos** cuando aparezcan: `df.head()`, `df.describe()`, `ds` (para xarray/NetCDF), `.shape`.
4. **Haz al menos un ejercicio** de la sección *Exercises* o de la columna de práctica de arriba.
5. **Escribe un resumen de 5 líneas** al final: problema, datos, modelo, métrica, una cosa que cambiarías.

---

## 3. Cada proyecto explicado

### 3.1 TC PRIMED: huracanes vistos por satélite (`tcprimed/`)

**La ciencia.** Los satélites con sensores de **microondas pasivas** miden la radiación que emite la Tierra a ciertas frecuencias (por ejemplo 37 y 89 GHz). Esa medida se expresa como **temperatura de brillo**. Las gotas y el hielo dentro de las nubes alteran esa señal: a 89 GHz el hielo de las tormentas fuertes "dispersa" la radiación y aparece frío, así que se ve claramente la estructura de lluvia de un huracán, incluso debajo de las nubes. TC PRIMED junta en un solo archivo por pasada del satélite: esas temperaturas de brillo, la lluvia estimada por el algoritmo **GPROF** de la NASA, imágenes infrarrojas, la trayectoria oficial del ciclón (*best track*) y el ambiente que lo rodea según el reanálisis **ERA5** (cizalladura del viento, temperatura, humedad). Son más de 197 000 pasadas de 2 300 ciclones entre 1998 y 2021.

**Los capítulos y el código.**

- **Cap. 0** es una introducción en texto.
- **Cap. 1a (NetCDF):** descarga un archivo con `requests.get(...)` y lo abre en memoria con `netCDF4.Dataset(nombre, memory=contenido)`. NetCDF es como una carpeta de arreglos: tiene **grupos** (subcarpetas), **variables** (arreglos con dimensiones) y **atributos** (metadatos como unidades). Aprendes a navegarlo.
- **Cap. 1b (Overpass file):** grafica las temperaturas de brillo de una pasada sobre el huracán Florence (2018), la lluvia de GPROF y la imagen infrarroja cercana en tiempo.
- **Cap. 1c (Environmental file):** usa `metpy` y `cartopy` para graficar la trayectoria, la cizalladura vertical del viento y un perfil vertical de temperatura y punto de rocío.
- **Cap. 2 (Composite analysis):** junta muchas pasadas del mismo huracán y promedia un parámetro a lo largo del tiempo para ver cómo evoluciona la tormenta.
- **Cap. 3 (Red neuronal por píxel):** el primer ML del proyecto. Para cada píxel, la entrada son las temperaturas de brillo a 36.64 y 89 GHz y la salida (el *target*) es la lluvia de GPROF en mm/h. Entrena con el huracán Jose, valida y prueba con otros huracanes de 2017. Normaliza las entradas con min-max, arma una red `Dense` con Keras y mira las curvas de pérdida (*loss*) de entrenamiento contra validación para detectar sobreajuste.
- **Cap. 4 (Regridding):** los datos de satélite vienen en una **franja (swath)** con píxeles irregulares. Con `pyresample` se pasan a una cuadrícula regular (proyección Mercator o Lambert) usando vecino más cercano o interpolación bilineal, para poder comparar sensores distintos.

**Idea clave para llevarte:** antes de hacer ML hay que entender el formato y la física de los datos. El cap. 3 funciona porque sabemos que 89 GHz "ve" el hielo de la lluvia.

### 3.2 Corrientes de resaca (`rip_current/`)

**La ciencia.** Una corriente de resaca (*rip current*) es un flujo angosto y fuerte de agua que sale de la playa hacia mar adentro. Es la principal causa de ahogamientos en playas de EE. UU. (unos 100 al año). Se forma cuando las olas rompen y el agua acumulada regresa por canales del fondo marino. Simularlo con física detallada es caro y requiere conocer la forma exacta del fondo (batimetría), que casi nunca se tiene. Por eso se plantea como un problema de **clasificación**: dadas las condiciones de olas y marea, ¿hay resaca peligrosa (sí/no)?

**Los datos.** Reportes de salvavidas de Mission Beach, San Diego (la "verdad"), olas de la boya NDBC 46258 (altura `WVHT`, periodo `DPD`/`APD`, dirección `MWD`) y nivel del agua de la estación de mareas 9410230 en La Jolla (`WL`). Se descargan juntos desde Zenodo.

**El código, en pasos.**

1. **Limpieza:** une los datos por fecha en un registro horario continuo.
2. **Ingeniería de variables:** con `lagmat` crea valores "retrasados" (lo que pasó hace 1, 2, 3 horas) y un índice de eventos `EP` que resume el oleaje de las últimas 72 horas.
3. **Visualización:** rosa de vientos (`windrose`) para la dirección de las olas y matrices de dispersión.
4. **Tres modelos, de menos a más:**
   - *Clase mayoritaria:* siempre dice "no hay resaca". Es la línea base; cualquier modelo útil debe superarla.
   - *Regresión logística:* da una probabilidad entre 0 y 1. Se elige un **umbral** para decidir "sí".
   - *Red neuronal densa:* 3 capas de 8 neuronas, salida `sigmoid`, evaluada con validación cruzada `RepeatedKFold`.
5. **Comparación:** matriz de confusión (aciertos y errores de cada tipo), exactitud y **curva ROC / AUC**.

**Idea clave:** cuando los eventos son raros (casi siempre "no"), la exactitud engaña. Por eso importan la línea base, el umbral y el AUC.

### 3.3 Potencia radiativa del fuego (`fire_radiative_power/`)

**La ciencia.** La **potencia radiativa del fuego (FRP)**, en megavatios, es la energía que emite un incendio y la detectan los satélites. Sirve para estimar cuánto humo se emite, lo que alimenta los pronósticos de calidad del aire. El problema es que la FRP observada tiene huecos (nubes, humo, ángulo del satélite). El objetivo es **predecir la FRP de la próxima hora** combinando la FRP observada antes con variables meteorológicas.

**Los datos.** **RAVE** (FRP horaria de satélite, en NetCDF) y **RAP** (modelo meteorológico de NOAA: temperatura, viento, ráfagas, humedad, precipitación, CAPE, humedad del suelo, radiación...). RAP se descarga con la librería **Herbie**, que trae archivos GRIB2 del modelo.

**El código.**

- **Parte 1 (opcional):** muestra cómo se construyó el dataset: descarga RAP con Herbie, lee RAVE y une ambos por ubicación y hora. Es lenta y baja bastante información.
- **Parte 2 (la importante):** descarga un CSV ya listo (2019 a 2021), elige 25 variables de entrada (`lat`, `lon`, `hour`, `temp2m`, `wind_gust`, ..., `24hr_roll_frp`) y la salida `rave_frp`. Separa 70 % entrenamiento y 30 % prueba, entrena un `RandomForestRegressor` y mide MAE y MSE. Al final grafica observado contra predicho.

**Qué es un Random Forest, simple:** un árbol de decisión hace preguntas tipo "¿la temperatura es mayor a 30 °C?" hasta llegar a una predicción. Un bosque entrena cientos de árboles, cada uno con una muestra distinta de los datos, y promedia sus respuestas. Funciona muy bien con tablas de datos y relaciones no lineales.

**Idea clave:** la variable `24hr_roll_frp` (promedio de FRP de las últimas 24 h) probablemente es la más importante; "lo que pasó recién" suele ser el mejor predictor de "lo que viene".

### 3.4 Perfiles de viento mar adentro (`offshore_wind_profiles/`)

**La ciencia.** Para planear parques eólicos marinos se necesita el viento a la altura de las turbinas (100 a 200 m), pero los satélites solo miden el viento cerca de la superficie (10 m). El viento cambia con la altura según la **estabilidad de la atmósfera**: si el aire está más frío que el mar, se mezcla mucho y el perfil es parejo; si está más caliente, la capa es estable y el viento aumenta fuerte con la altura. Por eso una entrada clave es `temp_diff = air_temp - SST` (aire menos temperatura del mar).

**Los datos.** Mediciones de boyas con **lidar flotante** (miden viento a varias alturas) como verdad, y el producto satelital **NOAA Blended Seawinds (NBSv2)** donde se aplica el modelo para crear el dataset final *NOAA Offshore Wind Profiles - USA*.

**El código.**

1. **Paso 1, ajustar hiperparámetros:** prueba combinaciones de `n_estimators` (número de árboles), `min_samples_leaf` y `max_features` dejando **una ubicación fuera** cada vez para validar. Así se mide si el modelo funciona en un lugar donde no entrenó.
2. **Paso 2:** entrena el modelo final con los parámetros óptimos del artículo (1000 árboles, `min_samples_leaf=30`, `max_features=1`).
3. **Paso 3:** lo prueba en una ubicación independiente (Atlantic Shores).
4. **Paso 4:** lo aplica a NBSv2 para generar mapas de viento en altura con `cartopy`.

**Idea clave:** validar dejando fuera ubicaciones completas (no filas al azar) es la forma honesta de saber si el modelo generaliza en el espacio.

### 3.5 Echofish: sonar de pesca en la nube (`echofish/`)

**La ciencia.** Los barcos científicos de NOAA llevan **ecosondas** (sonar) Simrad EK60 que emiten pulsos de sonido a varias frecuencias (18, 38, 70, 120 y 200 kHz) y escuchan el eco. Lo que regresa se mide como **Sv** (*volume backscattering strength*, en dB): cuánto sonido rebota en un volumen de agua. Peces, plancton y el fondo marino rebotan distinto según la frecuencia. Por ejemplo, los peces con vejiga natatoria reflejan fuerte a 38 kHz, mientras que el plancton pequeño responde más a 120-200 kHz. Restar el Sv de dos frecuencias (**diferencia de frecuencias**) ayuda a separar tipos de organismos. Una imagen de Sv contra tiempo y profundidad se llama **ecograma**.

**Los capítulos y el código.**

- **Cap. 1:** descarga archivos crudos `.raw` del bucket público `noaa-wcsd-pds` en AWS S3 con `boto3` (sin credenciales, modo `UNSIGNED`), los abre con `echopype.open_raw`, calcula y calibra Sv y grafica ecogramas por frecuencia.
- **Cap. 2:** en vez de bajar archivos, abre directamente desde S3 un **almacén Zarr** de nivel 2 (datos ya procesados de un crucero completo) con `s3fs` + `xarray`. Zarr guarda arreglos gigantes en trozos (*chunks*), así que solo descargas la parte que pides. Luego calcula diferencias como Sv(38 kHz) − Sv(18 kHz).
- **Cap. 3:** extrae latitud, longitud y tiempo del crucero, los convierte en un `GeoDataFrame` de **GeoPandas**, simplifica la trayectoria a una línea, la exporta como **GeoJSON** y la muestra en un mapa interactivo con `folium`.

**Idea clave:** con datos grandes en la nube no se descarga todo; se abre "en remoto" y se lee solo el pedazo que se necesita.

### 3.6 MagNet: tormentas geomagnéticas (`geomag-magnet/`)

**La ciencia.** El Sol emite un flujo continuo de partículas, el **viento solar**, que lleva su propio campo magnético. Cuando llega una ráfaga fuerte y su campo apunta "hacia el sur" (componente `bz` negativa), se acopla con el campo magnético de la Tierra y lo perturba: es una **tormenta geomagnética**. Eso puede dañar satélites, GPS y redes eléctricas. El índice **Dst** (en nanoteslas) mide qué tan perturbado está el campo cerca del ecuador: cerca de 0 es tranquilo, y −100 nT o menos es tormenta fuerte. El reto MagNet (2020) pedía predecir Dst de la hora actual y la siguiente usando datos del viento solar medidos por satélites entre el Sol y la Tierra, más el número de manchas solares.

**Los datos.** Unos 381 MB comprimidos (1.2 GB descomprimidos) desde Zenodo: `solar_wind.csv` (por minuto), `sunspots.csv`, `satellite_positions.csv` y `dst_labels.csv`. Hay datos *public* (para entrenar) y *private* (los que se ocultaron a los participantes, para evaluar).

**El código.**

- **Cap. 1 (desarrollar el LSTM):** explora los datos, agrega el viento solar por hora (media y desviación estándar), escala con `StandardScaler`, separa entrenamiento/validación/prueba por periodos de tiempo y arma secuencias de 32 horas con `timeseries_dataset_from_array`. Un **LSTM** es una red neuronal con "memoria" que lee la secuencia hora por hora y decide qué recordar. Al final guarda el modelo, el escalador y la configuración en `trained_models_lstm/`.
- **Cap. 2 (IA explicable, XAI):** carga un modelo guardado del cap. 1, mide la **importancia por permutación** (desordena una variable a la vez y ve cuánto empeora el error; si empeora mucho, esa variable importa) y analiza tormentas reales de los datos *private*, incluida una con Dst cercano a −180 nT.

**Idea clave:** en series de tiempo, el orden importa. Por eso se separan los datos por periodos (no al azar) y se usan modelos que leen secuencias.

---

## 4. Cómo correrlo en tu Windows

### 4.1 Qué quedó instalado

En la carpeta del repositorio hay un entorno virtual **`.venv`** con **Python 3.11** y todas las dependencias de los 6 proyectos (lista completa en [`requirements.txt`](requirements.txt)). También quedó registrado un kernel de Jupyter llamado **"Python (NOAA ML)"**.

Se eligió Python 3.11 y TensorFlow 2.15 a propósito: los notebooks de MagNet usan la API de Keras 2 (`batch_input_shape`, `keras.preprocessing`), que no existe en TensorFlow 2.16 o más nuevo (Keras 3). TensorFlow 2.15 solo funciona hasta Python 3.11.

Si algún día necesitas recrear el entorno desde cero, borra `.venv` y corre en PowerShell, dentro de la carpeta:

```bash
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

### 4.2 Abrir los notebooks

**Opción A, JupyterLab** (en PowerShell, dentro de la carpeta `learning-experiments`):

```bash
.\.venv\Scripts\activate
```

```bash
jupyter lab
```

Se abre en el navegador. Entra a la carpeta del proyecto, abre el `.ipynb` y verifica arriba a la derecha que el kernel sea **Python (NOAA ML)**.

**Opción B, VS Code:** abre la carpeta, abre un `.ipynb`, haz clic en *Select Kernel* (arriba a la derecha) y elige `.venv` o **Python (NOAA ML)**.

### 4.3 Notas por proyecto

Todos los notebooks se ejecutaron completos en tu máquina con este entorno (los tiempos son los medidos, con descargas incluidas):

| Proyecto | Cómo correrlo | Tiempo medido | Notas |
|---|---|---|---|
| `tcprimed` | Todas las celdas en orden | 15 s a 2.5 min; el cap. 2 ~6 min y el cap. 3 ~9 min | Descarga archivos `.nc` pequeños de S3 al vuelo. |
| `rip_current` | Todas las celdas | ~2 min | Descarga los datos de Zenodo en la carpeta del notebook. |
| `fire_radiative_power` | Parte 0 y luego salta a la Parte 2 | Parte 2: ~20 min (el Random Forest es lo lento) | La Parte 1 es opcional y descarga datos del modelo RAP (probé que Herbie descarga y lee RAP bien). La línea `import pygrib` de la Parte 1 dará error: **coméntala** (`# import pygrib`), no se usa. La Parte 2 empieza con `%reset`, que pregunta si confirmas: responde `y`. |
| `offshore_wind_profiles` | Todas las celdas | ~14 min (casi todo es el Paso 1, la búsqueda de hiperparámetros) | Si tienes prisa, reduce las listas de valores a probar. |
| `echofish` | Cap. 1, 2, 3 en orden | 20 s a 1.5 min cada uno | Acceso anónimo a S3, no necesitas cuenta de AWS. |
| `geomag-magnet` | Cap. 1 y luego cap. 2 | ~1.5 min cada uno con la configuración rápida; hasta 1.5 h con la del benchmark | Descarga ~381 MB a `data/` y usa varios GB de RAM: cierra programas pesados antes. En el cap. 1, elige la configuración de `model_config` (hay 4: la de "20 seconds" es para probar, la de "10-15 minutes" es un buen punto medio; la activa por defecto es la de 1.5 h). El cap. 2 te pide escribir el número del modelo guardado (un `input()`): escribe `0`. |

**Sobre la GPU:** tu RTX 3050 no se usará. Desde TensorFlow 2.11 no hay soporte de GPU en Windows nativo, así que todo corre en CPU. Para los tamaños de estos notebooks está bien. Si más adelante quieres GPU, la ruta es WSL2 (Linux dentro de Windows).

### 4.4 Problemas conocidos y cómo se resolvieron

- **`pygrib` no se instala en Windows** (no hay versión precompilada y pide compilador C++). El notebook de FRP lo importa pero no lo usa, así que basta con comentar esa línea.
Los notebooks fueron escritos para Google Colab (Linux, zona horaria UTC, versiones de 2023). Al correrlos en Windows aparecieron estos errores, y ya están **corregidos en tu copia** (cambios de 1 a 5 líneas cada uno):

| Notebook | Error | Corrección |
|---|---|---|
| `geomag-magnet` cap. 1 y 2 | `module 'urllib' has no attribute 'request'` | `import urllib.request` en vez de `import urllib` |
| `geomag-magnet` cap. 1 | `could not convert string to float: 'ac'` en `.corr()` | `.corr(numeric_only=True)`: pandas 2 ya no ignora solo las columnas de texto |
| `tcprimed` cap. 1c y 2 | `cannot select an axis to squeeze out` | Las fechas se convertían con la hora local de tu PC; ahora se usan en UTC, como los datos |
| `rip_current` y `offshore_wind_profiles` | `BadZipFile` al descargar | Zenodo ya no ofrece el zip completo; ahora se baja cada archivo del registro |
| `rip_current` | `Converting from datetime64 to int32` | En Windows `astype(int)` es de 32 bits; ahora es `astype('int64')` |

Además, en `requirements.txt` hay versiones fijadas a propósito: `numpy 1.26` y `tensorflow 2.15` (Keras 2), `matplotlib < 3.10` (la librería `toolbox` del notebook de FRP usa una función que se quitó en 3.10), `zarr < 3` y `xarray < 2025.7` (echopype 0.8.4 necesita zarr 2, y el xarray más nuevo ya no lo soporta), y `setuptools < 80` (echopype todavía usa `pkg_resources`). Si actualizas paquetes a mano, estas son las que pueden romper cosas.
- **Archivos descargados:** los notebooks guardan datos en su propia carpeta (por ejemplo `geomag-magnet/data/`, CSV en `rip_current/`). Están en `.gitignore` para no subirlos a git.

---

### Glosario rápido

- **Feature / variable de entrada:** lo que el modelo usa para predecir (temperatura, altura de ola...).
- **Target / label / etiqueta:** lo que se quiere predecir (lluvia, Dst, resaca sí/no).
- **Train / validation / test:** datos para aprender, para ajustar decisiones y para la evaluación final, respectivamente.
- **Overfitting (sobreajuste):** el modelo memoriza el entrenamiento y falla en datos nuevos. Se ve cuando la pérdida de validación sube mientras la de entrenamiento baja.
- **Regresión vs. clasificación:** predecir un número continuo vs. una categoría.
- **Hiperparámetro:** una perilla que eliges tú antes de entrenar (número de árboles, neuronas, épocas).
- **Normalizar / escalar:** poner todas las variables en rangos parecidos para que ninguna domine solo por su magnitud.
- **NetCDF / Zarr / GRIB2:** formatos para arreglos científicos grandes. NetCDF es un archivo; Zarr es un "directorio de trozos" ideal para la nube; GRIB2 es el formato de los modelos meteorológicos.
