# 🏆 Predictor del Mundial 2026 - Machine Learning

Este es un proyecto personal de Machine Learning desarrollado en Python para predecir el desarrollo y el ganador del **Mundial de la FIFA 2026**. Utiliza datos históricos de selecciones, rendimiento reciente y modelos basados en árboles de decisión para simular el torneo completo bajo su nuevo formato de 48 equipos.

---

## 🚀 Características del Proyecto

- **Algoritmo Principal:** `HistGradientBoostingClassifier` (Scikit-Learn).
- **Enfoque de Predicción:** El modelo predice la probabilidad del resultado de cada partido individual (Gana Local, Empate, Gana Visitante) en lugar de un campeón directo.
- **Simulador del Torneo:** Un script en Python que replica la fase de grupos del Mundial 2026 (12 grupos), calcula las tablas de posiciones dinámicamente y clasifica a las mejores selecciones a las rondas de eliminación directa.
- **Predictores Utilizados (Features):**
  - Rankings históricos (FIFA / Elo).
  - Porcentaje de victorias en el último año.
  - Historial de goles anotados y recibidos.

---

## 🛠️ Tecnologías y Herramientas

- **Lenguaje:** Python 3.11+
- **Librerías Clave:** `pandas`, `numpy`, `scikit-learn`
- **Entorno:** Visual Studio Code & Jupyter Notebooks
- **Control de Versiones:** Git & GitHub

---

## 📂 Estructura del Repositorio

```text
├── data/               # Archivos CSV con datos de partidos y rankings
├── notebooks/          # Jupyter Notebooks con el análisis exploratorio y entrenamiento
├── src/                # Scripts de Python (.py) para el motor de simulación
├── .gitignore          # Archivos excluidos del control de versiones (ej: venv/)
├── README.md           # Descripción del proyecto
└── requirements.txt    # Librerías necesarias para ejecutar el proyecto