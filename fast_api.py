from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import uvicorn

app = FastAPI()
model = joblib.load('obesity.pkl')

class UserInput(BaseModel):
    gender: int
    age: int
    height: int
    weight: int
    bmi: float

@app.get('/')
async def index():
    return {"message": "Welcome to the obesity prediction API!"}

@app.post('/predict')
async def predict(user_input: UserInput):
    prediction = model.predict([[user_input.gender, user_input.age, user_input.height, user_input.weight, user_input.bmi]])
    return JSONResponse(content={"prediction": prediction[0]})

if __name__ == '__main__':
    uvicorn.run(app, port=5012)
