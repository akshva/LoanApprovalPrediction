# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load data
df = pd.read_csv("loan_data.csv")

# Clean + encode (manual, consistent with API)
def map_bin(s, pos, neg):
    return s.str.strip().str.lower().map({pos: 1, neg: 0})

X = pd.DataFrame({
    "Gender":  map_bin(df["Gender"], "male", "female"),
    "Married": map_bin(df["Married"], "yes", "no"),
    "ApplicantIncome": df["ApplicantIncome"].astype(float),
    "LoanAmount":      df["LoanAmount"].astype(float),
})
y = df["Loan_Status"].str.strip().str.lower().map({"approved": 1, "rejected": 0}).astype(int)

# Split + train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "loan_model.pkl")
print("✅ Model trained and saved as loan_model.pkl")
