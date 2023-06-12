from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from lime import lime_tabular
import matplotlib.pyplot as plt

dataset = load_breast_cancer()
X = dataset.data
y = dataset.target
feature_names = dataset.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=29)

# model = RandomForestClassifier()
model = SVC(kernel='linear', probability=True, random_state=29)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
formatted_accuracy = "{:.2f}".format(accuracy_score(y_test, y_pred))
print(formatted_accuracy)

explainer = lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names, class_names=dataset.target_names,
                                              mode='classification', discretize_continuous=False)

sample_index = 40  # Index of the sample to explain
sample = X_test[sample_index]

explanation = explainer.explain_instance(sample, model.predict_proba, num_features=len(feature_names))

print('Predicted class:', dataset.target_names[model.predict(sample.reshape(1, -1))[0]])
print('True class:', dataset.target_names[y_test[sample_index]])

explanation.save_to_file('lime_results.html')
explanation.as_pyplot_figure()
plt.show()
