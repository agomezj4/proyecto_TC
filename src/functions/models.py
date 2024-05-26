from typing import Dict, Any, List

import pandas as pd
import numpy as np
import pickle
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn import set_config
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# 1. Performance modelos
def analyze_model_performance(
        results: Dict[str, Any],
        metric: str
) -> None:
    """
    Realiza el análisis estadístico para comparar los rendimientos de los modelos
    utilizando la prueba de Friedman y, si es necesario, la prueba post-hoc de Nemenyi.

    Parámetros:
        results (Dict[str, Any]): Diccionario con los resultados de cada modelo
        metric (str): Nombre de la métrica a analizar

    Retorna:
        None
    """
    # Obtener los nombres de los modelos y estadísticas
    ordered_results = order_models_by_performance(results, metric)
    model_names = ordered_results['ordered_models']
    statistics = ordered_results['statistics']
    logger.info(f"Comparando los siguientes modelos usando la métrica {metric}: {', '.join(model_names)}")

    # Preparar datos para Friedman Test
    scores = np.array([results[name][metric] for name in model_names])

    # Realizando la prueba de Friedman
    stat, p_value = friedmanchisquare(*scores)
    logger.info('Hipótesis nula: Los modelos tienen el mismo rendimiento.')
    logger.info(f"Estadístico de Friedman: {stat}\nP-value: {p_value}")

    # Explicación de los resultados de Friedman
    if p_value < 0.05:
        logger.info(f"Como el valor P es menor que 0.05 ({p_value}); se rechaza la hipótesis nula:\n"
                    f"Se encontraron diferencias estadísticamente significativas entre los modelos.")

        # Ejecutar la prueba post-hoc de Nemenyi
        scores_df = scores.T
        posthoc_results = sp.posthoc_nemenyi_friedman(scores_df)

        # Comprobar que posthoc_results es una matriz cuadrada y convertir a DataFrame
        if posthoc_results.shape[0] == posthoc_results.shape[1] and posthoc_results.shape[0] == len(model_names):
            summary_df = pd.DataFrame({
                'Mean': [statistics[name]['mean'] for name in model_names],
                'Median': [statistics[name]['median'] for name in model_names]
            }, index=model_names)
            logger.info(f"Resumen de la prueba post-hoc de Nemenyi para la métrica '{metric}':\n{summary_df}\n")
        else:
            logger.error("Error en los resultados de la prueba post-hoc: los resultados no tienen la forma esperada.")
    else:
        logger.info(f"No se encontraron diferencias estadísticamente significativas entre los modelos al evaluar la métrica '{metric}'.")

    return None



# 2. Ordenar modelos
def order_models_by_performance(
        results: Dict[str, Dict[str, np.ndarray]],
        metric: str
) -> Dict[str, Any]:
    """
    Ordena los modelos por su rendimiento en una métrica específica y calcula la media y mediana de los resultados.

    Parameters
    ----------
    results : Dict[str, Dict[str, np.ndarray]]
        Diccionario con los resultados de cada modelo
    metric : str
        Nombre de la métrica para ordenar los modelos

    Returns
    -------
    Dict[str, Any]
        Diccionario con el orden de los modelos, medias y medianas de los resultados.
    """
    statistics = {
        model: {
            'mean': np.mean(results[model][metric]),
            'median': np.median(results[model][metric])
        }
        for model in results
    }

    ordered_models = sorted(statistics.keys(), key=lambda x: statistics[x]['mean'], reverse=True)

    return {'ordered_models': ordered_models, 'statistics': statistics}


# 3. Entrenamiento y evaluación de modelos
def train_models_pd(
        df: pd.DataFrame,
        params: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    Entrena varios modelos de clasificación básicos y ensemble y evalúa su rendimiento usando validación cruzada.
    Además, realiza un análisis estadístico para comparar los rendimientos de los modelos.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de pandas con las características y la variable objetivo
    params: Dict[str, Any]
        Diccionario de parámetros models

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Diccionario donde las llaves son los nombres de los mejores modelos de cada categoría y los valores son los pickles y nombres.
    """
    logger.info("Inicio del entrenamiento y evaluación de modelos...\n")

    # Parámetros
    target = params['target']
    basic_models_params = params['basic_models']
    ensemble_models_params = params['ensemble_models']
    scoring = params['scoring']
    cv_config = params['cv_config']

    # Separar características y variable objetivo
    X = df.drop(columns=[target])
    y = df[target]

    # Definición del esquema de validación cruzada
    cv = RepeatedStratifiedKFold(**cv_config)

    # Entrenamiento y evaluación de modelos
    results = {'basic_models': {}, 'ensemble_models': {}}
    basic_models = {
        'Logistic Regression': LogisticRegression,
        'SVM': SVC,
        'KNN': KNeighborsClassifier,
        'Decision Tree': DecisionTreeClassifier
    }

    ensemble_models = {
        'Random Forest': RandomForestClassifier,
        'AdaBoost': AdaBoostClassifier,
        'XGBoost': XGBClassifier
    }

    # Mostrar el diagrama del pipeline en los logs
    set_config(display='diagram')

    # Función auxiliar para entrenar y evaluar modelos
    def train_and_evaluate(models, model_params, category):
        trained_models = {}
        for name, model_cls in models.items():
            model = model_cls(**model_params[name])
            pipeline = Pipeline(steps=[('classifier', model)])
            logger.info(f"Entrenando y evaluando el modelo: {name} ({category})")
            logger.info(f"{pipeline}\n")  # Mostrar el diagrama del pipeline
            cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False)
            model_results = {scoring_metric: cv_results[f'test_{scoring_metric}'] for scoring_metric in scoring.keys()}
            results[category][name] = model_results
            trained_models[name] = pipeline
        return trained_models

    # Entrenar y evaluar modelos básicos
    logger.info("Evaluando modelos básicos...\n")
    trained_basic_models = train_and_evaluate(basic_models, basic_models_params, 'basic_models')

    # Entrenar y evaluar modelos ensemble
    logger.info("Evaluando modelos ensemble...\n")
    trained_ensemble_models = train_and_evaluate(ensemble_models, ensemble_models_params, 'ensemble_models')

    # Realizar análisis de rendimiento del modelo
    for metric in scoring.keys():
        logger.info(f"Análisis de rendimiento para la métrica: {metric}")
        analyze_model_performance(results['basic_models'], metric)
        analyze_model_performance(results['ensemble_models'], metric)

    # Ordenar los modelos por rendimiento en la métrica principal
    ordered_basic_results = order_models_by_performance(results['basic_models'], scoring['recall'])  # Ajustar la métrica según sea necesario
    ordered_ensemble_results = order_models_by_performance(results['ensemble_models'], scoring['recall'])  # Ajustar la métrica según sea necesario

    # Seleccionar el mejor modelo básico y el mejor modelo ensemble y guardarlos como pickle
    best_models = {}
    best_basic_model = ordered_basic_results['ordered_models'][0]
    best_ensemble_model = ordered_ensemble_results['ordered_models'][0]

    # Guardar los mejores modelos como pickle
    best_models['basic'] = {
        'model_name': best_basic_model,
        'pickle': pickle.dumps(trained_basic_models[best_basic_model])
    }
    best_models['ensemble'] = {
        'model_name': best_ensemble_model,
        'pickle': pickle.dumps(trained_ensemble_models[best_ensemble_model])
    }

    logger.info(f"Los mejores modelos son: {best_basic_model} (básico) y {best_ensemble_model} (ensemble)\n")
    logger.info("Entrenamiento y evaluación de modelos completado!\n")

    return best_models
