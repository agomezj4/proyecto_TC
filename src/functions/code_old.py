from typing import Dict, List, Any, Tuple

import pandas as pd
import numpy as np
import unicodedata
import matplotlib.pyplot as plt  
import seaborn as sns
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import importlib
from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score, f1_score


from parameters.parameters import cum_cat, one_hot_encoder, rf_selection, ce_selection, features, hyperparameters






def cumulatively_categorise(
    column: pd.Series, 
    cum_cat: Dict[str, Any]
) -> pd.Series:
    """
    Categoriza acumulativamente una columna de un DataFrame de pandas, reemplazando los valores 
    que no cumplen con el umbral especificado

    Parameters
    ----------
    column : pandas.Series
        Columna de un DataFrame de pandas que se categorizará acumulativamente
    cum_cat: Dict[str, Any] 
         Diccionario de parámetros

    Returns
    -------
    pandas.Series
        Columna de un DataFrame de pandas con la categorización acumulativa aplicada
    """
    threshold = cum_cat['threshold']
    replacement_value = cum_cat['value']
    
    threshold_value = int(threshold * len(column))
    categories_list = []
    s = 0
    counts = Counter(column)
    
    # Categoriza acumulativamente la columna
    for i, j in counts.most_common():
        s += j  # Actualización para utilizar el conteo directamente
        categories_list.append(i)
        if s >= threshold_value:
            break
    
    categories_list.append(replacement_value)
    
    # Aplica la categorización acumulativa a la columna
    new_column = column.apply(lambda x: x if x in categories_list else replacement_value)

    return new_column



def one_hot_encoding(
    df: pd.DataFrame, 
    one_hot_encoder: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Aplica la función cumulatively_categorise a las columnas especificadas en el diccionario 
    de parámetros, luego aplica un One Hot Encoding a las columnas resultantes

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame de pandas al que se le aplicará One Hot Encoding.
    one_hot_encoder: Dict[str, Any] 
         Diccionario de parámetros

    Returns
    -------
    pd.DataFrame: DataFrame con las columnas transformadas.
    """

    for var in one_hot_encoder['columns']:
        # Aplica la función cumulatively_categorise a la columna
        df[var] = cumulatively_categorise(df[var], cum_cat)
        
        # Aplica One Hot Encoding a la columna resultante
        dummies = pd.get_dummies(df[var], prefix=var).astype(int)
        df = pd.concat([df, dummies], axis=1)
        df.drop(var, axis=1, inplace=True)

    return df



#caracteristicas
def add_random_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega dos variables aleatorias al DataFrame

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame al que se agregarán las variables aleatorias.

    Returns
    -------
    pd.DataFrame
        DataFrame con las variables aleatorias agregadas.
    """
    
    df["var_aleatoria_uniforme"] = np.random.rand(len(df))
    df["var_aleatoria_entera"] = np.random.randint(1, 5, size=len(df))
    
    return df



def random_forest_selection(
    df: pd.DataFrame, 
    rf_selection: Dict[str, Any]
) -> pd.DataFrame:
    """
    Entrena un modelo RandomForest y calcula la importancia de las características

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame de pandas al que se le calculará la importancia de las características
    rf_selection: Dict[str, Any]
        Diccionario de parámetros 

    Returns
    -------
    pd.DataFrame: DataFrame con la importancia de las características calculadas
    """
    
    # Divide los datos en conjuntos de entrenamiento y prueba
    X = df.drop(columns=rf_selection['target'])
    y = df[rf_selection['target']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=rf_selection['test_size'], random_state=rf_selection['seed'])


    # Agregar las variables aleatorias al DataFrame
    X_train = add_random_variables(X_train)

    # Crea un clasificador RandomForest
    clf = RandomForestClassifier(
        n_estimators=rf_selection['n_estimators'], 
        max_depth=rf_selection['max_depth'], 
        random_state=rf_selection['seed']
    )

    # Ajusta el modelo a tus datos de entrenamiento
    clf.fit(X_train, y_train)

    # Calcula la importancia de las características
    feature_importance = clf.feature_importances_

    # Obtén los nombres de las características
    feature_names = X_train.columns

    # Crea un DataFrame para mostrar la importancia de las características
    feature_importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": feature_importance}
    )

    # Ordena el DataFrame por importancia de manera descendente
    feature_importance_df = feature_importance_df.sort_values(
        by="Importance", ascending=False
    )

    # Obtener la importancia de las variables aleatorias
    random_var_imp_0_1 = feature_importance_df.loc[feature_importance_df["Feature"] == "var_aleatoria_uniforme", "Importance"].values[0]
    random_var_imp_1_4 = feature_importance_df.loc[feature_importance_df["Feature"] == "var_aleatoria_entera", "Importance"].values[0]

    # Eliminar las variables con importancia menor que las variables aleatorias
    feature_importance_df = feature_importance_df[(feature_importance_df["Importance"] > random_var_imp_0_1) & (feature_importance_df["Importance"] > random_var_imp_1_4)]

    return feature_importance_df



def entropy(p):
    """
    Calcula la entropía de un conjunto de probabilidades.

    Parameters
    ----------
    p : list
        Lista de probabilidades.

    Returns
    -------
    float: Valor de la entropía.
    """
    return -np.sum([pi * np.log2(pi) if pi > 0 else 0 for pi in p])



def conditional_entropy_selection(
    df: pd.DataFrame, 
    ce_selection: Dict[str, Any]
) -> pd.DataFrame:
    """
    Calcula la entropía condicional y la ganancia de información para un 
    conjunto de variables

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame de pandas al que se le calculará la entropía condicional
    ce_selection: Dict[str, Any] 
        Diccionario de parámetros

    Returns
    -------
    pd.DataFrame: DataFrame con la entropía condicional y la ganancia de 
    información calculadas
    """
    
    target = ce_selection['target']
    
    # Entropía variable objetivo
    des = df[target].value_counts()
    pr = des.values / des.values.sum()
    Ho = entropy(pr)
    
    # Cálculo de entropía condicional
    feature_names, feature_importance = [], []
    
    for columna in df.columns:
        if columna == target:
            continue
        H = 0
        feature_names.append(columna)
        
        for i in df[columna].unique():
            df_i = df[df[columna] == i]
            des = df_i[target].value_counts()
            pr = des.values / des.values.sum()
            Hcond = entropy(pr)
            prob = len(df_i) / len(df)
            H += Hcond * prob
            
        feature_importance.append(Ho - H)

    data_entropia = {'Feature': feature_names, 'Importance': feature_importance}
    
    df_entropia = pd.DataFrame(data_entropia)
    
    df_entropia.sort_values(by='Importance', ascending=False, inplace=True)

    return df_entropia



def intersect_top_features(
    df1: pd.DataFrame, 
    df2: pd.DataFrame,
    intersect_features: Dict[str, Any]
) -> set:
    """
    Obtiene las características más importantes de tres DataFrames basado en un diccionario de parámetros

    Parameters
    ----------
    df1, df2: pandas.DataFrame
        DataFrames de pandas de los que se obtendrán las características más importantes
    intersect_features: Dict[str, Any] 
        Diccionario de parámetros

    Returns
    -------
    set: Conjunto con las características más importantes.
    """
    # Obtén las n características más importantes del dataframe
    n = features['top_features']
    top_features_df1 = set(df1['Feature'][:n])
    top_features_df2 = set(df2['Feature'][:n])

    # Intersecta los conjuntos para obtener las características comunes a los tres dataframes
    top_features = top_features_df1.intersection(top_features_df2)

    # Retorna el conjunto con las características más importantes
    return top_features



#modelado
def evaluate_and_sort_models(
    models: List[Tuple[str, Any]], 
    X: pd.DataFrame, 
    y: pd.Series
) -> List[Tuple[str, float]]:
    """
    Evalúa y ordena los modelos basándose en la puntuación de validación cruzada.

    Parameters
    ----------
    models : List[Tuple[str, Any]]
        Lista de modelos a evaluar. Cada elemento de la lista es una tupla que contiene el nombre del modelo y el modelo en sí.
    X : pd.DataFrame
        DataFrame de características para la evaluación del modelo.
    y : pd.Series
        Series de etiquetas para la evaluación del modelo.

    Returns
    -------
    List[Tuple[str, float]]: Lista de tuplas que contiene el nombre del modelo y la puntuación media de validación cruzada.
    """

    # Inicializa una lista para almacenar las puntuaciones de los modelos
    model_scores = []

    # Evalúa cada modelo usando validación cruzada y almacena la puntuación media
    for name, model in models:
        score = cross_val_score(model, X, y, scoring=make_scorer(recall_score, average='macro'), cv=5)
        model_scores.append((name, np.mean(score)))

    # Ordena los modelos por puntuación en orden descendente
    model_scores.sort(key=lambda x: x[1], reverse=True)  # Ordenar por recall

    # Retorna la lista de modelos ordenados por puntuación
    return model_scores



def search_hyperparameters(
    X_train: pd.DataFrame, 
    Y_train: pd.Series,
    hyperparameters: Dict[str, Any]
) -> Dict[str, Tuple[Any, GridSearchCV]]:
    """
    Busca los mejores hiperparámetros para múltiples modelos dados utilizando GridSearchCV

    Parameters
    ----------
    X_train : pd.DataFrame
        DataFrame de características para la búsqueda de hiperparámetros.
    Y_train : pd.Series
        Series de etiquetas para la búsqueda de hiperparámetros.

    Returns
    -------
    Dict[str, Tuple[Any, GridSearchCV]]
        Diccionario con modelos y objetos GridSearchCV correspondientes.
    """
    best_models = {}

    for model_path in hyperparameters['model']:
        # Divide el path del modelo para obtener el módulo y el nombre de la clase
        module_name, class_name = model_path.rsplit('.', 1)

        # Importar el modelo dinámicamente
        ModelClass = getattr(importlib.import_module(module_name), class_name)

        # Inicializar el modelo
        model = ModelClass()

        # Inicializa GridSearchCV con los parámetros dados
        grid_search = GridSearchCV(
            model, 
            hyperparameters['param_grid'][class_name], 
            scoring=hyperparameters['scoring'], 
            refit=hyperparameters['refit'], 
            n_jobs=hyperparameters['n_jobs'], 
            cv=hyperparameters['cv']
        )

        # Ajusta GridSearchCV a los datos de entrenamiento
        grid_search.fit(X_train, Y_train)

        # Obtiene el mejor modelo después de la búsqueda de hiperparámetros
        best_model = grid_search.best_estimator_

        # Guarda el modelo y GridSearchCV en el diccionario
        best_models[class_name] = (best_model, grid_search)

    return best_models



def evaluation_metrics(
    optimization_results: Dict[str, Tuple[Any, Any]],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcula las métricas de evaluación y las predicciones para los resultados de optimización de varios modelos.
    """
    # DataFrames para almacenar los resultados
    metrics = pd.DataFrame(columns=['Modelo', 'precision', 'recall', 'balanced_accuracy', 'f1_score'])
    predictions_metrics = pd.DataFrame(columns=['Modelo', 'precision', 'recall', 'balanced_accuracy', 'f1_score'])

    # Iterar sobre cada modelo en los resultados de optimización
    for index, (model_name, (model, grid_search)) in enumerate(optimization_results.items(), start=1):
        # Extraer las métricas para el mejor conjunto de hiperparámetros
        metrics.loc[index] = [
            model_name,
            grid_search.cv_results_['mean_test_precision'][grid_search.best_index_],
            grid_search.cv_results_['mean_test_recall'][grid_search.best_index_],
            grid_search.cv_results_['mean_test_balanced_accuracy'][grid_search.best_index_],
            grid_search.cv_results_['mean_test_f1'][grid_search.best_index_]
        ]

        # Hacer predicciones y calcular las métricas para estas predicciones
        y_pred = model.predict(X_test)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')

        # Almacenar las métricas de predicciones en el DataFrame
        predictions_metrics.loc[index] = [model_name, precision, recall, balanced_accuracy, f1]

    # Ordenar los resultados de métricas
    metrics.sort_values(by='recall', ascending=True, inplace=True)

    return metrics, predictions_metrics


def matriz_confusion(
    modelo: Any, 
    X: pd.DataFrame, 
    y: pd.Series) -> plt.Figure:
    """
    Crea y visualiza una matriz de confusión para un modelo dado.

    Parameters
    ----------
    modelo : Any
        El modelo que se utilizará para hacer las predicciones.
    X_test : pd.DataFrame
        El conjunto de datos de prueba que se utilizará para hacer las predicciones.
    y_test : pd.Series
        Las etiquetas verdaderas para el conjunto de datos de prueba.

    Returns
    -------
    plt.Figure
        La figura de la matriz de confusión.
    """
    # Predicciones
    y_pred = modelo.predict(X)

    # Matriz de confusión
    cm = confusion_matrix(y_true=y, y_pred=y_pred, labels=None, normalize=None)

    # Etiquetas de las clases
    class_names = ['No Acepta TC', 'Acepta TC']

    # Crear figura y ejes
    fig, ax = plt.subplots()

    # Crear el mapa de calor
    heatmap = sns.heatmap(cm, cmap="Blues", annot=True, fmt="d", cbar=False,
                          xticklabels=class_names, yticklabels=class_names,
                          linewidths=0.5, linecolor='gray', ax=ax)

    # Añadir etiquetas a los ejes
    ax.set_xlabel('Predicciones del Modelo')
    ax.set_ylabel('Valores Reales')
    ax.set_title('Matriz de Confusión')

    # Ajustar la figura
    plt.tight_layout()

    # Devolver la figura
    return fig