import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# -------------------------------
# 📂 Load Dataset
# -------------------------------
data = pd.read_csv("crop_production.csv")

# Clean data
data = data.dropna()

# -------------------------------
# 📊 Prepare Data
# -------------------------------
data = pd.get_dummies(data)

X = data.drop("Production", axis=1)
y = data["Production"]

# -------------------------------
# 🌳 Train Decision Tree Model
# -------------------------------
model = DecisionTreeRegressor()
model.fit(X, y)

# -------------------------------
# 📊 Model Accuracy
# -------------------------------
accuracy = r2_score(y, model.predict(X))
print("Decision Tree Accuracy:", accuracy)

# -------------------------------
# 👤 User Input
# -------------------------------
try:
    area = float(input("\nEnter Area: "))
except:
    print("Invalid input! Please enter a number.")
    exit()

# Create sample input
sample = X.iloc[0:1].copy()
sample["Area"] = area

# Prediction
prediction = model.predict(sample)

print(f"Predicted Production (Decision Tree) for area {area} is {prediction[0]:.2f}")

# -------------------------------
# 📊 Scatter Graph
# -------------------------------
plt.scatter(data["Area"], data["Production"])
plt.xlabel("Area")
plt.ylabel("Production")
plt.title("Crop Production vs Area (Decision Tree)")
plt.show()

# -------------------------------
# 📊 Histogram
# -------------------------------
plt.hist(data["Production"])
plt.title("Production Distribution")
plt.xlabel("Production")
plt.ylabel("Frequency")
plt.show()