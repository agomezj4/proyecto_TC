# Parámetros para CumulativeCategorisation
cum_cat = {
    'threshold': 0.75,
    'value': 'otro'
}

#Parámetros para OneHotEncoding
one_hot_encoder= {
    'columns': [
        'cat__tipo_trabajo',
        'cat__estado_civil',
        'cat__educacion',
        'cat__mora',
        'cat__vivienda',
        'cat__consumo',
        'cat__contacto',
        'cat__mes',
        'cat__dia',
        'cat__resultado_anterior'

    ]
}

#Parámetros para random_forest_selection
rf_selection= {
    'target': 'y',
    'test_size': 0.3,
    'seed': 42,
    'n_estimators': 100,
    'max_depth': 4,
}

#Parámetros para conditional_entropy_selection
ce_selection= {
    'target': 'y'
}

#Parámetros para intersect_top_features
features= {
    'top_features': 20
}

#Parámetros para Logistic Regression
logistic_regression= {
    'C': 0.1,
    'solver': 'liblinear',
    'max_iter': 1000,
    'random_state': 42
}

#Parámetros para SVM
svc= {
    'C': 1,
    'kernel': 'rbf',
    'gamma': 'scale',
    'random_state': 42
}

#Parámetros para KNN
knn= {
    'n_neighbors': 5,
    'weights': 'uniform',
    'algorithm': 'auto',
    'leaf_size': 30
}

#Parámetros para Decision Tree
decision_tree= {
    'max_depth': 4,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42
}

#Parámetros para Random Forest
random_forest= {
    'n_estimators': 100,
    'max_depth': 4,
    'min_samples_split': 2,
    'random_state': 42
}

#Parámetros para AdaBoost
adaboost= {
    'n_estimators': 100,
    'learning_rate': 1,
    'random_state': 42
}

#Parámetros para Gradient Boosting
gradient_boosting= {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 4,
    'random_state': 42
}

#Parámetros para hiperparametros
hyperparameters= {  
    'model': [
        'sklearn.neighbors.KNeighborsClassifier',
        'sklearn.ensemble.AdaBoostClassifier',
        'sklearn.ensemble.GradientBoostingClassifier'
    ],
    
    'param_grid': {
        'KNeighborsClassifier': {
            'n_neighbors': [5, 10, 15],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [30, 40, 50]
        },
        'AdaBoostClassifier': {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.1, 1, 10],
        },
        'GradientBoostingClassifier': {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.1, 1, 10],
            'max_depth': [3, 4, 5]
        }
    },
    
    'scoring': [
        'balanced_accuracy',
        'precision',
        'recall',
        'f1'
    ],
    
    'cv': 5,

    'refit': 'recall',

    'n_jobs': -1

}