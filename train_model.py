from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model saved successfully")
