import pickle as pkl
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

scaler = pkl.load(open("scale.pkl", "rb"))
encoder = pkl.load(open("encode.pkl", "rb")) 
model = pkl.load(open("model.pkl", "rb"))
feature_order = pkl.load(open("features.pkl", "rb")) 

categorical_cols = ['cut', 'color', 'clarity']

class DiamondData(BaseModel):
    carat: float
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    x: float
    y: float
    z: float
    
@app.get('/')
async def index():
    return {'message': 'Predict diamond price'}

@app.post("/predict")
async def predict(data: DiamondData):
    data_dict = data.model_dump()
    sample = pd.DataFrame([data_dict])
    
    cats = sample[categorical_cols]
    encoded_cats = encoder.transform(cats)
    encoded_df = pd.DataFrame(
        encoded_cats, 
        columns=encoder.get_feature_names_out(categorical_cols), 
        index=sample.index
    )

    sample = sample.drop(columns=categorical_cols)
    sample = pd.concat([sample, encoded_df], axis=1)
    
    numeric_cols = [col for col in feature_order if col not in encoder.get_feature_names_out(categorical_cols)]
    sample[numeric_cols] = scaler.transform(sample[numeric_cols])

    sample = sample[feature_order]

    prediction = model.predict(sample)[0]
    return {
        "input_data": data_dict,
        "prediction": float(prediction)
    }
