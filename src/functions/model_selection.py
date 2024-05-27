from typing import Dict, Any

import pandas as pd
import numpy as np
import logging

from sklearn.pipeline import Pipeline

from bayes_opt import BayesianOptimization

from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from scipy.stats import ttest_ind

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


#1. Función para optimizar modelo KNN
def optimize_train_knn(
    model: Pipeline,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    params: Dict[str, Any]
) -> Pipeline:
    """
    Optimiza y entrena un KNeighborsClassifier usando optimización bayesiana para
    encontrar los mejores hiperparámetros basados en una métrica dada. Evalúa en el
    conjunto de validación y prueba.

    Parámetros
    ----------
    model : Pipeline
        Pipeline que contiene el modelo KNeighborsClassifier previamente entrenado

    df_val : pd.DataFrame
        Conjunto de datos de validación para la búsqueda de hiperparámetros

    df_test : pd.DataFrame
        Conjunto de datos de prueba para la evaluación final

    params: Dict[str, Any]
        Diccionario de parámetros model selection

    Retornos
    -------
    Pipeline
        Pipeline con el modelo KNeighborsClassifier entrenado con los mejores hiperparámetros encontrados.
    """

    # Extraer parámetros
    target_col = params["target_col"]
    seed = params["random_state"]
    target_metric = params["target_metric"]
    exploration_space = params["knn_classifier"]["exploration_space"]
    n_iter = params["knn_classifier"]["number_of_iterations"]
    init_points = params["knn_classifier"]["init_points"]
    cv_config = params["cv_config"]

    # Seleccionar variables
    X_val = df_val.drop(columns=[target_col])
    y_val = df_val[target_col].values
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col].values

    # Mapear opciones categóricas a índices numéricos
    weights_map = {i: v for i, v in enumerate(exploration_space["weights"])}
    metric_map = {i: v for i, v in enumerate(exploration_space["metric"])}

    def knn_evaluate(n_neighbors, p, weights, metric):
        # Convertir índices numéricos de vuelta a opciones categóricas
        weights = weights_map[int(weights)]
        metric = metric_map[int(metric)]

        # Utilizar el modelo pasado como parámetro y actualizar sus hiperparámetros
        model.set_params(
            classifier__n_neighbors=int(n_neighbors),
            classifier__p=int(p),
            classifier__weights=weights,
            classifier__metric=metric
        )
        cv = RepeatedStratifiedKFold(**cv_config)
        scores = cross_val_score(model, X_val, y_val, cv=cv, scoring=target_metric)
        return np.mean(scores)

    # Actualizar exploration_space para usar índices numéricos
    exploration_space_numeric = {
        "n_neighbors": exploration_space["n_neighbors"],
        "p": exploration_space["p"],
        "weights": (0, len(weights_map) - 1),
        "metric": (0, len(metric_map) - 1)
    }

    # Configurar y ejecutar la optimización bayesiana
    optimizer = BayesianOptimization(
        f=knn_evaluate, pbounds=exploration_space_numeric, random_state=seed, verbose=2
    )
    optimizer.maximize(n_iter=n_iter, init_points=init_points)

    # Recopilar los resultados
    best_params = optimizer.max["params"]
    best_score = optimizer.max["target"]

    logger.info(f"Mejores parámetros KNN: {best_params} con {best_score} para {target_metric}")

    # Convertir índices numéricos de vuelta a opciones categóricas
    best_params["weights"] = weights_map[int(best_params["weights"])]
    best_params["metric"] = metric_map[int(best_params["metric"])]

    # Actualizar el modelo con los mejores hiperparámetros
    model.set_params(
        classifier__n_neighbors=int(best_params["n_neighbors"]),
        classifier__p=int(best_params["p"]),
        classifier__weights=best_params["weights"],
        classifier__metric=best_params["metric"]
    )

    # Evaluar el modelo optimizado en el conjunto de prueba
    cv = RepeatedStratifiedKFold(**cv_config)
    test_score = cross_val_score(model, X_test, y_test, cv=cv, scoring=target_metric)
    logger.info(f"Desempeño KNN con el conjunto test: {test_score}")

    return model


#2. Función para optimizar modelo XGBoost
def optimize_train_xgboost(
    model: Pipeline,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    params: Dict[str, Any]
) -> Pipeline:
    """
    Optimiza y entrena un XGBoost classifier usando optimización bayesiana para
    encontrar los mejores hiperparámetros basados en una métrica dada. Evalúa en el
    conjunto de validación y prueba.

    Parámetros
    ----------
    model : Pipeline
        Pipeline que contiene el modelo XGBClassifier previamente entrenado

    df_val : pd.DataFrame
        Conjunto de datos de validación para la búsqueda de hiperparámetros

    df_test : pd.DataFrame
        Conjunto de datos de prueba para la evaluación final

    params: Dict[str, Any]
        Diccionario de parámetros model selection

    Retornos
    -------
    Pipeline
        Pipeline con el modelo XGBClassifier entrenado con los mejores hiperparámetros encontrados.
    """

    # Extraer parámetros
    target_col = params["target_col"]
    seed = params["random_state"]
    metric = params["target_metric"]
    exploration_space = params["xgboost_classifier"]["exploration_space"]
    n_iter = params["xgboost_classifier"]["number_of_iterations"]
    init_points = params["xgboost_classifier"]["init_points"]
    cv_config = params["cv_config"]

    # Seleccionar variables
    X_val = df_val.drop(columns=[target_col])
    y_val = df_val[target_col].values
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col].values

    def xgboost_evaluate(max_depth, n_estimators, learning_rate, min_child_weight, subsample, colsample_bytree, gamma):
        # Convertir parámetros a int donde sea necesario
        max_depth = int(max_depth)
        n_estimators = int(n_estimators)
        min_child_weight = int(min_child_weight)

        # Establecer los parámetros en el modelo
        model.set_params(
            classifier__max_depth=max_depth,
            classifier__n_estimators=n_estimators,
            classifier__learning_rate=learning_rate,
            classifier__min_child_weight=min_child_weight,
            classifier__subsample=subsample,
            classifier__colsample_bytree=colsample_bytree,
            classifier__gamma=gamma
        )

        cv = RepeatedStratifiedKFold(**cv_config)
        scores = cross_val_score(model, X_val, y_val, cv=cv, scoring=metric)
        return np.mean(scores)

    # Configurar y ejecutar la optimización bayesiana
    optimizer = BayesianOptimization(
        f=xgboost_evaluate, pbounds=exploration_space, random_state=seed, verbose=2
    )
    optimizer.maximize(n_iter=n_iter, init_points=init_points)

    # Recopilar los resultados
    best_params = optimizer.max["params"]
    best_score = optimizer.max["target"]

    logger.info(f"Mejores parámetros XGBoost: {best_params} con {best_score} para {metric}")

    # Actualizar el modelo con los mejores hiperparámetros
    model.set_params(
        classifier__max_depth=int(best_params["max_depth"]),
        classifier__n_estimators=int(best_params["n_estimators"]),
        classifier__learning_rate=best_params["learning_rate"],
        classifier__min_child_weight=int(best_params["min_child_weight"]),
        classifier__subsample=best_params["subsample"],
        classifier__colsample_bytree=best_params["colsample_bytree"],
        classifier__gamma=best_params["gamma"]
    )

    # Evaluar el modelo optimizado en el conjunto de prueba
    cv = RepeatedStratifiedKFold(**cv_config)
    test_score = cross_val_score(model, X_test, y_test, cv=cv, scoring=metric)
    logger.info(f"Desempeño XGBoost con el conjunto test: {test_score}")

    return model


#3. Selección del mejor modelo optimizado
def ab_test_models(
        model_a: Pipeline,
        model_b: Pipeline,
        df_test: pd.DataFrame,
        params: Dict[str, Any]
) -> Pipeline or None:
    """
    Realiza un A/B test para comparar dos modelos usando una métrica específica y devuelve el mejor modelo.

    Parámetros
    ----------
    model_a : Pipeline
        Primer modelo optimizado.
    model_b : Pipeline
        Segundo modelo optimizado.
    df_test : pd.DataFrame
        Conjunto de datos de prueba para evaluar el rendimiento de los modelos
    params: Dict[str, Any]
        Diccionario de parámetros model selection

    Retornos
    -------
    Pipeline
        El modelo con mejor rendimiento basado en la prueba estadística.
    """
    # Parámetros
    target_col = params["target_col"]
    metric = params["target_metric"]
    cv_config = params["cv_config"]

    # Obtener los nombres de los modelos desde los Pipelines
    classifier_step_model_a = next(step for name, step in model_a.steps if name == 'classifier')
    model_a_name = classifier_step_model_a.__class__.__name__

    classifier_step_model_b = next(step for name, step in model_b.steps if name == 'classifier')
    model_b_name = classifier_step_model_b.__class__.__name__

    logger.info(f"Realizando A/B test entre los modelos {model_a_name} y {model_b_name} usando la métrica {metric}")

    # Seleccionar variables
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col].values

    # Configurar la validación cruzada
    cv = RepeatedStratifiedKFold(**cv_config)

    # Obtener las puntuaciones de validación cruzada para ambos modelos
    scores_a = cross_val_score(model_a, X_test, y_test, cv=cv, scoring=metric)
    scores_b = cross_val_score(model_b, X_test, y_test, cv=cv, scoring=metric)

    logger.info(f"Scores del modelo {model_a_name}: {scores_a}")
    logger.info(f"Scores del modelo {model_b_name}: {scores_b}")

    # Realizar la prueba t de Student
    t_stat, p_value = ttest_ind(scores_a, scores_b)

    logger.info(f"t-statistic: {t_stat}, p-value: {p_value}")

    # Determinar el mejor modelo basado en la prueba estadística
    if p_value < 0.05:
        if scores_a.mean() > scores_b.mean():
            logger.info(f"El modelo A es significativamente mejor que el modelo {model_b_name}")
            return model_a
        else:
            logger.info(f"El modelo B es significativamente mejor que el modelo {model_a_name}")
            return model_b
    else:
        logger.info(f"No hay una diferencia significativa entre los modelos {model_a_name} y {model_b_name}")
        return None

