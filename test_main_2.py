 #mainurl.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import requests

app = FastAPI()

# Load the saved model
#model = tf.keras.models.load_model('csa_glaucoma1.h5')

# Define the request payload using Pydantic BaseModel
# class ImagePayload(BaseModel):
#     image_url: str


@app.get("/")
def foobar():
    return {
        "foo":"bar"
    }
# Endpoint to make predictions0
# import requests

# url = "https://glaucoma-detection-api-52f531d65dd6.herokuapp.com/predict"
# files = {"file": open("glauc True 1.png", 'rb')}
# res = requests.post(url, files=files)


# Endpoint to make predictions
@app.get("/")
async def predict_image():
        
        # Load the saved model
        model = tf.keras.models.load_model('csa_glaucoma1.h5')
    
        pic = "glaucoma false.png"
        # Download and preprocess the image
        img = image.load_img(pic, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize pixel values to [0, 1]

        # Make predictions using te loaded model
        prediction = model.predict(img_array)
        result = True if prediction[0][0] > 0.5 else False 
        # print(prediction)
        return {"Glaucoma": result,
                "value": float(prediction[0][0])}
        #testing api
        # return {"test":"Hello World!",
                # "Data": test_var
                # }
