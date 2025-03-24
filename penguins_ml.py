# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# import data
penguin_df = pd.read_csv('penguins.csv')

# Data Cleaning
penguin_df.dropna(inplace=True)

# Feature Engineering & Selections
output = penguin_df['species']
features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm',
                       'flipper_length_mm', 'body_mass_g', 'sex']]
features = pd.get_dummies(features)
output, uniques = pd.factorize(output)

# Train_test split
X_train, X_test, y_train, y_test = train_test_split(features, output, test_size=0.8, random_state = 42)

# Model Training and Evaluation
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train.values, y_train)
y_pred = rfc.predict(X_test.values)
score = accuracy_score(y_pred, y_test)
print('The accuracy of the model is {}'.format(score))

# Saving model
rf_pickle = open('random_forest_pickle', 'wb')
pickle.dump(rfc, rf_pickle)
rf_pickle.close()
output_pickle = open('output_penguin.pickle', 'wb')
pickle.dump(uniques, output_pickle)
output_pickle.close()

fig, ax = plt.subplots()
ax = sns.barplot(x=rfc.feature_importances_, y=features.columns)
plt.title('Features importance for species prediction')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
fig.savefig('feature_importance.png')

