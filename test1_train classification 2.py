import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# =========================
# STEP 1: Load and prepare training data
# =========================
train = pd.read_csv("D:/Learn Machine Learning/datasets/Projects/titanic/train.csv")

# Drop unnecessary columns
train = train.drop(["PassengerId", "Name"], axis=1)

# Drop rows with missing values (simple approach)
train = train.dropna()

# Separate object and numeric columns
object_data = train.select_dtypes(include="object")
num_data = train.select_dtypes(exclude="object")

# Encode categorical features
label_enc = LabelEncoder()
for col in object_data.columns:
    object_data[col] = label_enc.fit_transform(object_data[col])

# Combine back into one DataFrame
train_data = pd.concat([num_data, object_data], axis=1)

# Features (X) and Target (y)
X = train_data.drop("Survived", axis=1)
y = train_data["Survived"]

# =========================
# STEP 2: Train-test split
# =========================
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.95, random_state=42)

# =========================
# STEP 3: Train model
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

print("Train accuracy:", model.score(x_train, y_train))
print("Test accuracy:", model.score(x_test, y_test))

# =========================
# STEP 4: Prepare test data
# =========================
test = pd.read_csv("D:/Learn Machine Learning/datasets/Projects/titanic/test.csv")
ids = test["PassengerId"]

# Drop same columns as training
test = test.drop(["PassengerId", "Name"], axis=1, errors="ignore")

# Fill missing numeric values with mean
test = test.fillna(test.mean(numeric_only=True))

# Fill missing categorical values with "Missing" and encode
for col in test.select_dtypes(include="object").columns:
    test[col] = test[col].fillna("Missing")
    test[col] = label_enc.fit_transform(test[col].astype(str))

# Ensure same columns as training set
for col in X.columns:
    if col not in test.columns:
        test[col] = 0
test = test[X.columns]

# =========================
# STEP 5: Predict survival
# =========================
predictions = model.predict(test)

# =========================
# STEP 6: Save results
# =========================
output = pd.DataFrame({
    "PassengerId": ids,
    "Survived": predictions
})
output.to_csv("Survived.csv", index=False)
print("Predictions saved to 'Survived.csv'")
