# app.py (or in a dedicated router module)

from fastapi import FastAPI, HTTPException
from schemas import TrainRequest, TrainResponse
from model import train_model  # a function we will define for training logic
from config import MODEL_FILE_PATH  # path to save the model (e.g., "models/latest_model.pkl")

app = FastAPI(title="ML Model API")

@app.post("/train", response_model=TrainResponse)
def train(data: TrainRequest):
    """
    Train a model with the given dataset.
    """
    # Convert the list of DataPoint into features X and labels y for training
    try:
        X = [dp.features for dp in data.data]
        y = [dp.label for dp in data.data]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid training data format: {e}")
    
    # Train the model using custom logic (e.g., a function in model.py)
    model = train_model(X, y)
    # Save the trained model to disk
    import pickle
    with open(MODEL_FILE_PATH, "wb") as f:
        pickle.dump(model, f)
    
    # Return response with message and model path
    return {"message": "Model trained successfully", "model_path": MODEL_FILE_PATH}
