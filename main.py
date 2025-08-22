# main.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory=".")

# Load trained model
model = joblib.load("loan_model.pkl")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    # Serve the index.html from the same folder
    return templates.TemplateResponse("index.html", {"request": request})

# Make all params optional so hitting /predict by mistake won't 422
@app.get("/predict")
def predict(
    Gender: str | None = None,
    Married: str | None = None,
    ApplicantIncome: float | None = None,
    LoanAmount: float | None = None,
):
    # Basic validation
    if None in (Gender, Married, ApplicantIncome, LoanAmount):
        return {"error": "Missing parameters. Use the form on '/' to submit."}

    # Encode like training
    gender = 1 if str(Gender).strip().lower() == "male" else 0
    married = 1 if str(Married).strip().lower() == "yes" else 0

    x = np.array([[gender, married, float(ApplicantIncome), float(LoanAmount)]])
    pred = int(model.predict(x)[0])
    result = "Approved" if pred == 1 else "Rejected"
    return {"loan_status": result}
