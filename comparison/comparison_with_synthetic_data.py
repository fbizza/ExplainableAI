from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import utils
import numpy as np

n_samples = 1000
n_features = 5
true_coefficients = np.array([13.0, -22.5, 16.5, 0.0, 0.0])
X, y, features_names = utils.generate_synthetic_dataset(n_samples, n_features, true_coefficients)

unique, counts = np.unique(y, return_counts=True)
print("Distribution of y:")
for label, count in zip(unique, counts):
    print(f"Class {label}: {count} samples")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)

#clf = SVC()

clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

formatted_accuracy = "{:.2f}".format(accuracy_score(y_test, y_pred))
print("Accuracy: ", formatted_accuracy)

perm_importance = permutation_importance(clf, X, y, n_repeats=5, random_state=42)

sorted_idx = perm_importance.importances_mean.argsort()

plt.barh(range(len(features_names)), perm_importance.importances_mean[sorted_idx], align='center',
         color=['red' if i == sorted_idx[-1] else 'blue' for i in range(len(features_names))])
plt.yticks(range(len(features_names)), [features_names[i] for i in sorted_idx])
plt.xlabel('Permutation Importance')
plt.ylabel('Features')
plt.title('Features Importance Permutation')
plt.show()

