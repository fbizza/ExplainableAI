from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from lime import lime_tabular
import matplotlib.pyplot as plt

dataset = fetch_california_housing()
X = dataset.data
y = dataset.target
feature_names = dataset.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=29)

model = RandomForestRegressor()
model.fit(X_train, y_train)

explainer = lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names, class_names=['House Value'],
                                              mode='regression', discretize_continuous=False)

sample_index = 15  # Index of the sample to explain
sample = X_test[sample_index]

explanation = explainer.explain_instance(sample, model.predict, num_features=len(feature_names))

print('Predicted house value:', model.predict(sample.reshape(1, -1))[0])
print('True house value:', y_test[sample_index])

# Plot the explanation
explanation.save_to_file('lime_results.html')
explanation.as_pyplot_figure()
plt.show()