#  Predictor de partidos de selecciones de fútbol masculino - ML

Este es un proyecto personal de Machine Learning desarrollado en Python para predecir partidos de fútbol de selecciones. Utiliza datos históricos de selecciones, rendimiento reciente y un modelo de Gradient Tree Boosting para predecir probabilidades del resultado de un partido de fútbol

---

##  Características del Proyecto

- **Enfoque de Predicción:** El modelo predice la probabilidad del resultado de cada partido individual (Gana Local, Empate, Gana Visitante) 
- **Predictores Utilizados (Features):**
  * Diferencia de ELO y Ranking FIFA.
  * Diferencia de puntos FIFA.
  * Diferencia de goles a favor y en contra (media de los últimos 3 partidos).
  * Diferencil de puntos obtenidos en el rendimiento reciente.
---


## Rendimiento y Métricas

El modelo actual ha sido evaluado con un conjunto de datos de prueba desde enero de 2024 hasta junio de 2026, logrando resultados competitivos para la predicción deportiva:
* **Log Loss:** 0.8350 
* **ROC AUC:** * Victoria Local: 0.83
  * Victoria Visitante: 0.85
  * Empate: 0.65
* **Precision general:** 60.14%
    
## Herramientas

- **Lenguaje:** Python 3.11+
- **Librerías Clave:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`
- **Modelo Principal:** `HistGradientBoostingClassifier`
- **Entorno:** Visual Studio Code & Jupyter Notebooks
- **Control de Versiones:** Git & GitHub

---

## Estructura del Proyecto

```text
├── data/                   # Datasets
├── plots/                  # Gráficos de evaluación (Matriz de confusión, Curvas ROC)
├── notebooks/              #algunas pruebas y gráficas del dataset final                      
├── preprocess.py            # Limpieza y generación de predictores
├── train.py                # Entrenamiento del modelo
├── model_GB.joblib         # Modelo final serializado y listo para producción
├── requirements.txt        # Dependencias del proyecto
└── README.md               # Documentación
```
## Datos
El dataset completo no está incluido en este repositorio debido a su tamaño. Para obtenerlo:
1. Descarga el archivo desde [https://drive.google.com/file/d/1VZzPtMxREr1ZAmnTouREOjDx6tK5yQX7/view?usp=sharing].
2. Guárdalo en la carpeta `/data/` como `processed_data_last_3.csv`.

