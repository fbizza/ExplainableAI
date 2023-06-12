from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import utils

X, y, features_names = utils.import_compas_dataset()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

formatted_accuracy = "{:.2f}".format(accuracy_score(y_test, y_pred))
print("Accuracy: ", formatted_accuracy)

importance = clf.feature_importances_
plt.barh(range(len(importance)), importance, align='center')
plt.yticks(range(len(importance)), features_names)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Decision Tree Feature Importances")
plt.show()

perm_importance = permutation_importance(clf, X, y, n_repeats=10)

sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(range(len(features_names)), perm_importance.importances_mean[sorted_idx], align='center')
plt.yticks(range(len(features_names)), [features_names[i] for i in sorted_idx])
plt.xlabel('Permutation Importance')
plt.ylabel('Features')
plt.title('Feature Permutation Importances')
plt.show()