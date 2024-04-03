from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np
from io import BytesIO
import tensorflow as tf

app = FastAPI()

# Load the saved model
model = tf.keras.models.load_model('small_model_pipeline.pkl')

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Check if the uploaded file is an image
    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=415, detail="Unsupported media type")

    # Read the image file
    contents = await file.read()
    image = Image.open(BytesIO(contents))

    # Preprocess the image
    image = image.resize((150, 150))  # Resize to match model input size
    img_array = np.array(image) / 255.0  # Convert to numpy array and normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make predictions using the loaded model
    prediction = model.predict(img_array)
    result = True if prediction[0][0] > 0.5 else False

    return {
        "Glaucoma": result,
        "Confidence": float(prediction[0][0])
    }
