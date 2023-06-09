from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import utils

X, y, features_names = utils.import_boston_dataset()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

clf = RandomForestRegressor()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

importance = clf.feature_importances_
plt.barh(range(len(importance)), importance, align='center')
plt.yticks(range(len(importance)), features_names)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Decision Tree Feature Importances")
plt.show()