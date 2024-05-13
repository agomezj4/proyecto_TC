from typing import Dict, Any

import pandas as pd
import numpy as np
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# 1. Performance modelos
def analyze_model_performance(results, metric):
    """
    Realiza el análisis estadístico para comparar los rendimientos de los modelos
    utilizando la prueba de Friedman y, si es necesario, la prueba post-hoc de Nemenyi.

    Parámetros:
        results (dict): Diccionario con los resultados de cada modelo.
        metric (str): Nombre de la métrica a analizar.

    Retorna:
        None
    """
    # Preparar datos para Friedman Test
    scores = np.array([results[name][metric] for name in results])

    # Realizando la prueba de Friedman
    stat, p_value = friedmanchisquare(*scores)
    logger.info('Hipótesis nula: Los modelos tienen el mismo rendimiento.')
    logger.info(f"Estadístico de Friedman: {stat}, P-value: {p_value}")

    # Explicación de los resultados de Friedman
    if p_value < 0.05:
        logger.info(f"Como valor P es menor que 0.05 {p_value}, se rechaza la hipótesis nula.")
        logger.info("Se encontraron diferencias estadísticamente significativas entre los modelos.")
        # Ejecutar la prueba post-hoc de Nemenyi
        scores_df = np.array([results[name][metric] for name in results]).T
        posthoc_results = sp.posthoc_nemenyi_friedman(scores_df)
        logger.info("Resultados de la prueba post-hoc de Nemenyi:\n" + str(posthoc_results))
    else:
        logger.info("No se encontraron diferencias estadísticamente significativas entre los modelos.")

    return None


# 2. Entrenamiento de modelos base
def train_basic_models_pd(
        df: pd.DataFrame,
        params: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """
    Entrena varios modelos de clasificación básicos y evalúa su rendimiento usando validación cruzada.
    Además, realiza un análisis estadístico para comparar los rendimientos de los modelos.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de pandas con las características y la variable objetivo.
    params: Dict[str, Any]
        Diccionario de parámetros models.

    Returns
    -------
    Dict[str, np.ndarray]
        Diccionario con los nombres de los modelos como claves y los resultados de las métricas de evaluación como valores.
    """
    logger.info("Inicio del entrenamiento y evaluación de modelos básicos...")

    # Parámetros
    target = params['target']
    models = params['basic_models']
    scoring = params['scoring']
    cv_config = params['cv_config']

    # Separar características y variable objetivo
    X = df.drop(columns=[target])
    y = df[target]

    # Definición del esquema de validación cruzada
    cv = RepeatedStratifiedKFold(**cv_config)

    # Entrenamiento y evaluación de modelos
    results = {}
    model_registry = {
        'Logistic Regression': LogisticRegression,
        'SVM': SVC,
        'KNN': KNeighborsClassifier,
        'Decision Tree': DecisionTreeClassifier
    }

    # Entrenar y evaluar cada modelo
    for name, model_cls in model_registry.items():
        model = model_cls(**models[name])
        pipeline = Pipeline(steps=[('classifier', model)])
        logger.info(f"Entrenando y evaluando el modelo: {name}")
        cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False)
        model_results = {metric: cv_results[f'test_{metric}'] for metric in scoring.keys()}
        results[name] = model_results

    return results