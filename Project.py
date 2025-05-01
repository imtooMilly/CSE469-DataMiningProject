import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Step 1: Load dataset
df = pd.read_csv("pet_adoption_data.csv")  

# Step 2: Preprocess the data
# Separate features and target
df = df.drop(columns=["PetID"])
X = df.drop(columns=["AdoptionLikelihood"])  # 'adopted' should be 0 (not adopted) or 1 (adopted)
y = df["AdoptionLikelihood"]

# Convert categorical variables to dummy/one-hot encoding
X = pd.get_dummies(X)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train the decision tree
model = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Step 6: Visualize the tree
plt.figure(figsize=(30, 10))
plot_tree(
    model,
    filled=True,
    feature_names=X.columns.tolist(),
    class_names=["Not Adopted", "Adopted"],
    rounded=True
)
plt.title("Decision Tree for Animal Adoption Prediction")
plt.show()
