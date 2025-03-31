import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay, ConfusionMatrixDisplay, roc_curve, auc, balanced_accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier 
import time
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline


# https://www.geeksforgeeks.org/random-forest-hyperparameter-tuning-in-python/
# https://stackoverflow.com/questions/50245684/using-smote-with-gridsearchcv-in-scikit-learn

df = pd.read_csv('data/raw/processed_features_full.csv')

y = df['Risk_Flag']
X = df.drop('Risk_Flag',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1) 

start = time.time()
# Define the hyperparameter search space
param_dist = {
    'rf__n_estimators': [50,100,150],
    'rf__max_depth': [None, 3,5,7,10],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4],
    'rf__max_features': ['sqrt', 'log2', None],
}

pipeline = Pipeline([('smote', SMOTE(random_state=1)), ('rf', RandomForestClassifier(random_state=1))])
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, 
                                   n_iter=10, cv=cv, verbose=2, random_state=1, n_jobs=-1,scoring='recall')


random_search.fit(X_train, y_train)
end = time.time()
print(f"Best parameters found: {random_search.best_params_}")

best_model = random_search.best_estimator_
ypred = best_model.predict(X_test)

metrics = pd.DataFrame([{
        "Accuracy": accuracy_score(y_test, ypred),
        "balanced_accuracy": balanced_accuracy_score(y_test, ypred),
        "precision": precision_score(y_test,ypred),
        "recall": recall_score(y_test,ypred),
        "f1_score": f1_score(y_test,ypred),
        "roc_auc": roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]),
        "conf_matrix": confusion_matrix(y_test, ypred),
        "time (s)": end-start
    }])
conf_matrix =  confusion_matrix(y_test, ypred)
print(metrics)
print(conf_matrix)


cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix)
cm_display.plot()
plt.title('Confusion Matrix - Hypertuned RF')
plt.savefig(f'results/figures/conf_matrix_tuned.png')
plt.close()

fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name='tuned rf')
display.plot()
plt.title('ROC Curve - Hypertuned RF')
plt.savefig(f'results/figures/roc_tuned.png')
plt.close()
metrics.to_csv('data/raw/tuned_metrics.csv',index=False)
print(f"took {end-start}")
