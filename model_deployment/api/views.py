from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.core.files.storage import default_storage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os 
from django.conf import settings

############

import sys
print(os.listdir("../"))
sys.path.append("../")
sys.path.append("./")

print("ASdas" ,os.listdir())


from src.inference import predict

#############


# Load your trained model (assuming it is a TensorFlow/Keras model)
MODEL_PATH = "models/prediction_model.h5"
model = load_model(MODEL_PATH)

def image_upload(request):
    prediction = None
    image_url = None
    request.session.flush()

    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        
        # Define the path to save the file
        temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
        os.makedirs(temp_dir, exist_ok=True)  # Ensure the 'temp/' directory exists

        # Save the file to the temp directory
        file_path = default_storage.save(os.path.join('temp', uploaded_file.name), uploaded_file)

        # Full path to the saved file
        full_file_path = os.path.join(settings.MEDIA_ROOT, file_path)
        image_url = f"{settings.MEDIA_URL}{file_path}"  # URL for accessing the image
        print("IMAGE_URL " ,image_url)
        # Preprocess the image
        # img = load_img(full_file_path, target_size=(224, 224))  # Adjust size to model's input
        # img_array = img_to_array(img)
        # img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        # img_array = img_array / 255.0  # Normalize if required

        # Predict with the model
        # prediction = model.predict(img_array)
        # prediction = np.argmax(prediction, axis=1)  # Example for classification models
        prediction = predict(model , full_file_path)



    return render(request, 'upload.html', {'prediction': prediction, 'image_url': image_url})



