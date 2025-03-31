import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt

# https://guhanesvar.medium.com/feature-selection-based-on-mutual-information-gain-for-classification-and-regression-d0f86ea5262a
# https://stats.stackexchange.com/questions/321970/imbalanced-data-smote-and-feature-selection
# Should NOT use smote prior to feature selection

df = pd.read_csv('data/raw/processed_data.csv')
y = df['Risk_Flag']
X = df.drop(columns=['Risk_Flag'])

mi = mutual_info_classif(X, y, random_state=1)

mi_df = pd.DataFrame({
    'Feature': X.columns,
    'Mutual Info': mi
})

mi_df = mi_df.sort_values(by='Mutual Info', ascending=False)

# Plot the mutual information scores
plt.figure(figsize=(10, 8))
plt.barh(mi_df['Feature'], mi_df['Mutual Info'])
plt.xlabel('MI Score')
plt.ylabel('Features')
plt.title('MI Scores of features')
plt.savefig('results/figures/mutual_info', bbox_inches='tight')

selected_features = mi_df['Feature'].head(6).values
selected_features_with_target = list(selected_features) + ['Risk_Flag']
new_df = df[selected_features_with_target]
new_df.to_csv('data/raw/processed_features_full.csv',index=False)

print(mi_df)
