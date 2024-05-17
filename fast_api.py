from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import uvicorn

app = FastAPI()

# Load the model
model = joblib.load('obesity.pkl')


app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origins=['*']
)

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
    uvicorn.run(app, host="0.0.0.0", port=5012)
