import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator 
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from IPython.display import display
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    make_scorer, 
    get_scorer, 
)

def calculate_metrics_cv(model:BaseEstimator, X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.DataFrame, y_test:pd.DataFrame, model_name:str, scoring:dict=None, cv:int=5) -> dict:
    """
    Entrena un modelo de clasificación, realiza predicciones y calcula métricas de evaluación usando validación cruzada.

    Input:
    - model (BaseEstimator): El modelo de clasificación que se va a entrenar y evaluar.
    - X_train (pd.DataFrame): Conjunto de características para entrenamiento.
    - X_test (pd.DataFrame): Conjunto de características para prueba.
    - y_train (pd.Series): Etiquetas del conjunto de entrenamiento.
    - y_test (pd.Series): Etiquetas del conjunto de prueba.
    - model_name (str): Nombre del modelo para identificar los resultados en la salida.
    - scoring (dict, opcional): Diccionario de métricas personalizadas; si no se especifica, se usa un conjunto predeterminado de métricas.
      Las métricas predeterminadas incluyen 'accuracy', 'precision', 'recall', 'f1' y 'roc_auc'.
    - cv (int, opcional): Número de pliegues para la validación cruzada (default=5).

    Returns:
    - results: Diccionario que contiene el nombre del modelo y las métricas calculadas tanto en validación cruzada 
               como en el conjunto de prueba, incluyendo 'accuracy', 'precision', 'recall', 'f1', y 'roc_auc'.

    """

    # Definir las métricas de evaluación si no se pasaron como argumento
    if scoring is None:
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, zero_division=0),
            'recall': make_scorer(recall_score, zero_division=0),
            'f1': make_scorer(f1_score, zero_division=0),
            'roc_auc': 'roc_auc'
        }

    # Validación cruzada usando cv (default= 5), 
    cv_results = cross_validate(
        model, 
        X_train, 
        y_train, 
        cv=cv,
        scoring=scoring,
        return_train_score=True
    )
    
    # Obtener promedio de validación cruzada
    results = {"Method": model_name}
    for key in cv_results:
        if key.startswith('test_'):
            metric_name = key[5:].capitalize()
            results[metric_name] = cv_results[key].mean()
    
    # Entrenar el modelo en todo el set de entrenamiento
    model.fit(X_train, y_train)
    
    # Hacer predicciones usando el test de prueba
    y_pred = model.predict(X_test)
    
    # Métricas usando el test de prueba
    for metric_name in scoring:
        display_metric_name = metric_name.replace('_', ' ').capitalize()
        if metric_name == 'roc_auc':
            # Verificar si el modelo incluye predict_proba
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                # Manejar tanto el caso binario como el multiclase
                if y_proba.shape[1] == 2:
                    y_proba = y_proba[:, 1]  # Usar las probabilidades para la clase positiva
                    results["Test_" + display_metric_name] = roc_auc_score(y_test, y_proba)
                else:
                    # Caso multiclase
                    results["Test_" + display_metric_name] = roc_auc_score(
                        y_test, y_proba, multi_class='ovr', average='weighted'
                    )
            else:
                results["Test_" + display_metric_name] = None
        else:
            # Obtener la métrica especificada por el scorer
            scorer = get_scorer(metric_name)
            # Obtener la métrica especificada por el scorer
            score = scorer._score_func(y_test, y_pred, **scorer._kwargs)
            results['Test_' + display_metric_name] = score

    return results

def create_pipelines(models: dict, samplers: dict = None, feature_selection: dict = None, scalers: dict = None) -> list:
    """
    Crea múltiples pipelines combinando modelos, técnicas de muestreo, selección de características y escaladores.
    
    Input:
    - models (dict): Diccionario con modelos de clasificación, donde la clave es el nombre del modelo y el valor es el objeto del modelo.
    - samplers (dict, opcional): Diccionario con técnicas de muestreo (ej. SMOTE), donde la clave es el nombre y el valor es el objeto del sampler.
      Si no se especifica, no se usa un método de balanceo.
    - feature_selection (dict, opcional): Diccionario con métodos de selección de características, donde la clave es el nombre y el valor 
      es una función que toma un modelo y devuelve un selector de características. Si no se especifica, no se usa un selector.
    - scalers (dict, opcional): Diccionario con escaladores, donde la clave es el nombre y el valor es el objeto del escalador (ej. StandardScaler).
      Si no se especifica, no se usa un escalador.

    Output:
    - pipelines_list (list): Lista de diccionarios, donde cada uno contiene un pipeline configurado y su nombre.
      Cada pipeline puede incluir pasos como escalado, muestreo, selección de características y clasificación.

    """
    # Si samplers es None o esta vacío, crear un diccionario predeterminado con 'None'
    if not samplers:
        samplers = {'None': None}

    # Si feature_selection es None o esta vacío, crear un diccionario predeterminado con 'None'
    if not feature_selection:
        feature_selection = {'None': lambda model: None}
    
    # Si scalers es None o vacío, crear un diccionario predeterminado con 'None'
    if not scalers:
        scalers = {'None': None}

    pipelines_list = []

    # Iterar sobre todas las combinaciones de modelos, muestreadores, escaladores y selectores de características
    for model_name, model in models.items():
        for sampler_name, sampler in samplers.items():
            for scaler_name, scaler in scalers.items():
                for fs_name, fs_function in feature_selection.items():
                    # Crear el nombre del pipeline combinando los nombres de los componentes utilizados
                    pipeline_name_parts = [model_name]

                    if sampler_name != 'None':
                        pipeline_name_parts.append(sampler_name)

                    if scaler_name != 'None':
                        pipeline_name_parts.append(scaler_name)

                    if fs_name != 'None':
                        pipeline_name_parts.append(fs_name)

                    pipeline_name = ' - '.join(pipeline_name_parts)

                    # Crear los pasos del pipeline
                    steps = []

                    if scaler_name != 'None' and scaler is not None:
                        steps.append(('scaler', scaler))

                    if sampler_name != 'None' and sampler is not None:
                        steps.append(('sampler', sampler))

                    # Generar el selector de características usando la función proporcionada
                    feature_selector = fs_function(model)

                    if fs_name != 'None' and feature_selector is not None:
                        steps.append(('feature_selection', feature_selector))

                    # Añadir el modelo de clasificación al pipeline
                    steps.append(('classifier', model))

                    # Crear el pipeline adecuado dependiendo de si se usa un sampler
                    if sampler_name != 'None' and sampler is not None:
                        # Usar ImbPipeline si hay un sampler
                        pipeline = ImbPipeline(steps)
                    else:
                        # Usar Pipeline estándar si no hay sampler
                        pipeline = Pipeline(steps)

                    pipelines_list.append({'pipeline': pipeline, 'name': pipeline_name})

    return pipelines_list

def evaluate_pipelines(X: pd.DataFrame, y: pd.Series, pipelines_list: list) -> list:
    """
    Evalúa una lista de pipelines en un conjunto de datos especificado, utilizando validación cruzada
    y el conjunto de prueba para calcular las métricas de rendimiento.

    Input:
    - X: DataFrame con las características del conjunto de datos.
    - y: Series con las etiquetas o variable objetivo del conjunto de datos.
    - pipelines_list: Lista de diccionarios que contienen los pipelines configurados y sus nombres.

    Output:
    - Una lista de diccionarios, donde cada uno contiene el nombre del pipeline evaluado y las métricas calculadas
      como precisión, recall, F1-score, AUC-ROC, entre otras.

    """
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    results = []

    # Iterar sobre cada pipeline en la lista proporcionada
    for pipeline_info in pipelines_list:
        pipeline = pipeline_info['pipeline']  # Obtener el pipeline
        model_name = pipeline_info['name']  # Obtener el nombre del modelo

        # Evaluar el pipeline utilizando la función calculate_metrics_cv para calcular las métricas
        metrics = calculate_metrics_cv(
            model=pipeline,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            model_name=model_name
        )
        
        # Agregar las métricas obtenidas a los resultados
        results.append(metrics)
    
    return results


def display_metrics(results: list) -> None:
    """
    Convierte los resultados de las métricas de evaluación en DataFrames separados para las métricas 
    de validación cruzada y de prueba, y los muestra en pantalla.

    Input:
    - results: Lista de diccionarios que contienen los nombres de los modelos y las métricas calculadas 
      (como precisión, recall, F1-score, AUC-ROC, etc.) tanto para validación cruzada como para el conjunto de prueba.

    """
    
    # Convertir los resultados en un DataFrame
    metrics_df = pd.DataFrame(results)
    metrics_df.set_index("Method", inplace=True)

    # Separar las métricas de validación cruzada (que no tienen el prefijo 'Test_')
    cv_metrics = metrics_df.filter(regex='^(?!Test_).*', axis=1)
    # Cambiar 'Roc_auc' por 'AUC' para estandarizar los nombres de las métricas
    cv_metrics.columns = cv_metrics.columns.str.replace('Roc_auc', 'AUC')
    cv_metrics.reset_index(inplace=True)

    # Separar las métricas calculadas en el conjunto de prueba (que tienen el prefijo 'Test_')
    test_metrics = metrics_df.filter(regex='^Test_', axis=1)
    # Remover el prefijo 'Test_' y cambiar 'Roc auc' por 'AUC' para estandarizar
    test_metrics.columns = test_metrics.columns.str.replace('^Test_', '', regex=True)
    test_metrics.columns = test_metrics.columns.str.replace('Roc auc', 'AUC')
    test_metrics.reset_index(inplace=True)

    # Mostrar las métricas de validación cruzada
    print("Cross-Validation Metrics:")
    display(cv_metrics)

    # Mostrar las métricas del conjunto de prueba
    print("\nTest Set Metrics:")
    display(test_metrics)
