
from fastapi import FastAPI
from app.model_registry import load_model

app = FastAPI()

@app.get("/ocr")
def ocr(path: str):
    model = load_model("paddleocr_model")
    return {"text": model.predict(path)}

@app.get("/classify")
def classify(text: str):
    model = load_model("specialty_classifier")
    return {"prediction": model.predict([text])[0]}

@app.get("/recommend")
def recommend(text: str):
    model = load_model("cosine_recommender")
    return {"recommendations": model.predict([text])}
