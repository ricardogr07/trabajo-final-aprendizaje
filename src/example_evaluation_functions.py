from evaluation_functions import *
import pandas as pd

from sklearn.linear_model import LogisticRegression  

import faraway.datasets.prostate as prostate

prostate_data = prostate.load()

X = prostate_data.drop(columns=['svi'])
y = prostate_data['svi']

# Samplers
samplers = {
    'None': None
}

# Modelos
models = {
    'Logistic Regression': LogisticRegression(fit_intercept=False,class_weight='balanced', max_iter=1000, random_state=42),
}

pipelines_list = create_pipelines(models, samplers)

# Evaluar los pipelines
results = evaluate_pipelines(X, y, pipelines_list)

display_metrics(results)