from evaluation_functions import *
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier  
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier 
import faraway.datasets.prostate as prostate

prostate_data = prostate.load()

X = prostate_data.drop(columns=['svi'])
y = prostate_data['svi']


# Modelos
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVC': SVC(probability=True),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

pipelines_list = create_pipelines(models)

# Evaluar los pipelines
results = evaluate_pipelines(X, y, pipelines_list)

display_metrics(results)
