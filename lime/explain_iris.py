from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from lime import lime_tabular
import matplotlib.pyplot as plt

data = load_iris()
X = data.data
y = data.target
feature_names = data.feature_names
class_names = data.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

explainer = lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names, class_names=class_names)

# Choose a data point from the test set to explain
test_instance = X_test[0]


explanation = explainer.explain_instance(test_instance, model.predict_proba, num_features=3)


print("Explanation for the prediction:")
print(explanation.as_list())

explanation.save_to_file('lime_results.html')
explanation.as_pyplot_figure()
plt.show()
