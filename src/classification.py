import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay, ConfusionMatrixDisplay, roc_curve, auc, balanced_accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import time
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold

# https://towardsdatascience.com/the-right-way-of-using-smote-with-cross-validation-92a8d09d00c7
# https://stats.stackexchange.com/questions/638852/is-it-really-so-bad-to-do-smote-on-the-training-set-before-crossvalidation

class ModelInfo:

    def __init__(self,classifier,X,y,isSmote, state=1):

        if (isSmote):
            self.classifier = Pipeline([
                ('smote', SMOTE(random_state=state)),
                ('classifier', classifier)
            ])
        else:
            self.classifier = classifier

        self.isSmote = isSmote
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.20, random_state=state)
        self.state = state
        self.results = {}
        self.time_elapsed = 0

    def train_and_test(self, cv, scoring='balanced_accuracy'):

        start = time.time()

        # Use same fold each time so we can reproduce results
        folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.state)
        cv_scores = cross_val_score(self.classifier, self.X_train, self.y_train, cv=folds, scoring=scoring)
        self.classifier.fit(self.X_train, self.y_train)
        y_pred = self.classifier.predict(self.X_test)
        pred_prob = self.classifier.predict_proba(self.X_test)[:, 1]

        self.results = {
            "Avg_cv_score": np.mean(cv_scores),
            "y_pred": y_pred,
            "pred_prob": pred_prob,
            "accuracy": accuracy_score(self.y_test, y_pred),
            "f1_score": f1_score(self.y_test,y_pred),
            "recall": recall_score(self.y_test,y_pred),
            "balanced_accuracy": balanced_accuracy_score(self.y_test, y_pred),
            "precision": precision_score(self.y_test,y_pred),
            "roc_auc": roc_auc_score(self.y_test, pred_prob),
            "conf_matrix": confusion_matrix(self.y_test, y_pred)
        }

        end = time.time()
     
        self.time_elapsed = end-start
        print(f"{type(self.classifier).__name__} took {self.time_elapsed}")

    def plot_metrics(self, type):
        cm_display = ConfusionMatrixDisplay(confusion_matrix = self.results['conf_matrix'])
        cm_display.plot()
        plt.title(f'Confusion Matrix - {type}')
        plt.savefig(f'results/figures/conf_matrix_{type}.png')
        plt.close()

        fpr, tpr, _ = roc_curve(self.y_test, self.results['pred_prob'])
        roc_auc = auc(fpr, tpr)
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name=type)
        display.plot()
        plt.title(f'ROC Curve - {type}')
        plt.savefig(f'results/figures/roc_{type}.png')
        plt.close()
    
def add_model_results(df, model, name):

    new_row = pd.DataFrame([{
        'Model': name,
        'SMOTE Used': model.isSmote,
        'Avg CV Score (Bal. Acc.)': model.results['Avg_cv_score'],
        'Accuracy': model.results['accuracy'],
        'Balanced Accuracy': model.results['balanced_accuracy'],
        'Precision': model.results['precision'],
        'Recall': model.results['recall'],
        'F1-Score': model.results['f1_score'],
        'AUC-ROC': model.results['roc_auc'],
        'Time (s)': model.time_elapsed
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    return df

    
df = pd.read_csv('data/raw/processed_features_full.csv')
no_selection = pd.read_csv('data/raw/processed_data.csv')

metrics_df = pd.DataFrame(columns=[
    'Model', 'SMOTE Used', 'Avg CV Score (Bal. Acc.)', 'Accuracy','Balanced Accuracy', 'Precision', 'Recall', 
    'F1-Score', 'AUC-ROC', 'Time (s)'
])

metrics_df = metrics_df.astype({
    'Model': 'str', 'SMOTE Used': 'str', 'Avg CV Score (Bal. Acc.)': 'float64', 'Accuracy': 'float64', 'Balanced Accuracy': 'float64',
    'Precision': 'float64', 'Recall': 'float64', 'F1-Score': 'float64', 'AUC-ROC': 'float64', 'Time (s)': 'float64'
})

y = df['Risk_Flag']
X = df.drop('Risk_Flag',axis=1)

y_ns = no_selection['Risk_Flag']
X_ns = no_selection.drop('Risk_Flag', axis=1)

## Random Forest Models

rf_feature_selected = ModelInfo(RandomForestClassifier(random_state=1), X, y, False)
rf_feature_selected.train_and_test(cv=5)
metrics_df = add_model_results(metrics_df,rf_feature_selected,"Random Forest (Feature Selection)")

rf_all_features = ModelInfo(RandomForestClassifier(random_state=1), X_ns, y_ns, False)
rf_all_features.train_and_test(cv=5)
metrics_df = add_model_results(metrics_df,rf_all_features,"Random Forest (All Variables)")

rf_feature_selected_smote = ModelInfo(RandomForestClassifier(random_state=1), X, y, True)
rf_feature_selected_smote.train_and_test(cv=5)
metrics_df = add_model_results(metrics_df,rf_feature_selected_smote,"Random Forest (Feature Selection)")

rf_all_features_smote = ModelInfo(RandomForestClassifier(random_state=1), X_ns, y_ns, True)
rf_all_features_smote.train_and_test(cv=5)
metrics_df = add_model_results(metrics_df,rf_all_features_smote,"Random Forest (All Variables)")

## XGBoost Models

xgboost_feature_selected = ModelInfo(XGBClassifier(random_state=1), X, y, False)
xgboost_feature_selected.train_and_test(cv=5)
metrics_df = add_model_results(metrics_df,xgboost_feature_selected,"XGBoost (Feature Selection)")

xgboost_all_features = ModelInfo(XGBClassifier(random_state=1), X_ns, y_ns, False)
xgboost_all_features.train_and_test(cv=5)
metrics_df = add_model_results(metrics_df,xgboost_all_features,"XGBoost (All Variables)")

xgboost_feature_selected_smote = ModelInfo(XGBClassifier(random_state=1), X, y, True)
xgboost_feature_selected_smote.train_and_test(cv=5)
metrics_df = add_model_results(metrics_df,xgboost_feature_selected_smote,"XGBoost (Feature Selection)")

xgboost_all_features_smote = ModelInfo(XGBClassifier(random_state=1), X_ns, y_ns, True)
xgboost_all_features_smote.train_and_test(cv=5)
metrics_df = add_model_results(metrics_df,xgboost_all_features_smote,"XGBoost (All Variables)")

print(metrics_df)
metrics_df.to_csv('data/raw/classification_metrics.csv',index=False)

xgboost_feature_selected_smote.plot_metrics('xgboost_fs')
xgboost_all_features_smote.plot_metrics('xgboost_alldata')
rf_feature_selected_smote.plot_metrics('random_forest_fs')
rf_all_features_smote.plot_metrics('random_forest_alldata')