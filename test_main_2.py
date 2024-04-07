from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np
from io import BytesIO
import pickle

app = FastAPI()

# Load the saved model
model = pickle.load(open('small_model_pipeline.pkl', 'rb'))
scaler = pickle.load(open('Glaucoma_scaker.pkl', 'rb'))

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Check if the uploaded file is an image
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=415, detail="Unsupported media type")

    # Read the image file
    contents = await file.read()
    image = Image.open(BytesIO(contents))

    # Preprocess the image
    image = image.resize((150, 150))  # Resize to match model input size
    img_array = np.array(image)  # Convert to numpy array
    img_array = img_array.reshape(1, -1)  # Flatten the image
    img_array = scaler.transform(img_array)  # Scale the image data

    # Make predictions using the loaded model
    prediction = model.predict(img_array)
    result = True if prediction[0][0] > 0.5 else False

    return {
        "Glaucoma": result,
        "Confidence": float(prediction[0][0])
    }
