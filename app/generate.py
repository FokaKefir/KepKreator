from fastapi import FastAPI, Depends, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
import re
import numpy as np
from models import CGAN
import os
from PIL import Image

imgs_path = os.path.join(os.getcwd(), 'images')
if not os.path.exists(imgs_path):
      os.makedirs(imgs_path)

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


cgan_mnist = CGAN(latent_dim=100, n_classes=10)
cgan_mnist.load_models('models/tmp-model/generator.h5', 'models/tmp-model/discriminator.h5')

print("Ready to go!")

# TODO: write EMNIST CGAN 
#cgan_emnist = CGAN(latent_dim=100, n_classes=10)

def generate_and_save_image(cgan: CGAN, text: str, input: np.ndarray) -> str:
    # create the path to the image
    path = os.path.join(imgs_path, f"{text}.jpg")

    # generate images
    images = cgan.generate_images(input)

    # concate images
    num_imgs, h, w, ch = images.shape
    concatenated_img = np.zeros((h, num_imgs * w, ch))
    for i, img in enumerate(images):
        concatenated_img[:, i*w:(i+1)*w] = img
    concatenated_img = concatenated_img.reshape(h, num_imgs * w)

    # create final numpy image which is prepared to be saved 
    final_img = (concatenated_img[:, :] + 1) / 2 * 255.0
    final_img = final_img.astype('uint8') 

    # save image
    img = Image.fromarray(final_img, mode="L")
    img.save(path)
    
    # return path
    return path

def validate_text_mnist(text: str = Path(...)):
    regex=r'^[0-9]+$'
    if not re.match(regex, text):
        raise HTTPException(status_code=422, detail="The input must contain numbers only")
    return text

def validate_text_emnist(text: str = Path(...)):
    regex=r'^[A-Za-z0-9_]+$'
    if not re.match(regex, text):
        raise HTTPException(status_code=422, detail="The input doesn't match with the allowed characters")
    return text

@app.get("/generate/mnist/{text}")
def generate_mnist_image(text: str = Depends(validate_text_mnist)):
    nums = np.array(list(text), dtype=int).reshape(-1, 1)
    path = generate_and_save_image(cgan_mnist, text, nums)
    return {"text": text, "path": path}

@app.get("/generate/emnist/{text}")
def generate_emnist_image(text: str = Depends(validate_text_emnist)):
    nums = np.array(list(text), dtype=int).reshape(-1, 1)
    


    return {"text": text}