from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
app=FastAPI()
model=joblib.load("models/student_model.pkl")
@app.get("/")
def home():
    return{"message":"welcome to the student performance prediction api running"}
class studentData(BaseModel):
    study_hour:float
    attendence:float
    previous_marks:float
@app.post("/predict")
def predict(student:studentData):
    try:
        input_data = pd.DataFrame([{
            "study_hours": student.study_hours,
            "attendance": student.attendance,
            "previous_marks": student.previous_marks
        }])
        
        prediction = model.predict(input_data)
        return {"predicted_result": str(prediction[0])}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

