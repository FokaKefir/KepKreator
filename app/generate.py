# Import dependencies
from fastapi import FastAPI, Depends, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
import re
import numpy as np
from models import CGAN
import os
from PIL import Image

# Create folder for saving the images
imgs_path = os.path.join(os.getcwd(), 'images')
if not os.path.exists(imgs_path):
      os.makedirs(imgs_path)

# Create API
app = FastAPI()

# add CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load CGAN which generates numbers 
cgan_num = CGAN(latent_dim=96, n_classes=10)
cgan_num.load_models(
    'models/num-model/generator.h5', 
    'models/num-model/discriminator.h5'
)

# load CGAN which generates letters
cgan_letter = CGAN(latent_dim=64, n_classes=26)
cgan_letter.load_models(
    'models/letter-model-zs/generator.h5',
    'models/letter-model-zs/discriminator.h5'
)

# The models are loaded!
print("Ready to go!")

# validate prompt
def validate_text(text: str = Path(...)):
    regex=r'^[A-Za-z0-9]+$'
    if not re.match(regex, text):
        raise HTTPException(status_code=422, detail="The input doesn't match with the allowed characters")
    return text

# encode character to number which is acceptable for the model
def encode_letter(letter: str):
  return ord(letter) - 97

# returns the labels from the text
def get_labels(text: str) -> (np.ndarray, np.ndarray):
    # set lowercase the letters
    text = text.lower()

    # create placeholders for the labels
    arr_letters = []
    arr_nums = []

    # go through every character
    for c in text:
        # check if it's a number
        if c in '0123456789':
            arr_nums.append(int(c))
        # otherwise it's a letter
        else:
            arr_letters.append(encode_letter(c))

    # transform placeholder to numpy arrays
    letter_labels = np.array(arr_letters).reshape(-1, 1)
    num_labels = np.array(arr_nums).reshape(-1, 1)

    # return values
    return num_labels, letter_labels

# generates images using the cgan model and then transforms the outputs
def generate_and_shape_images(cgan: CGAN, input: np.ndarray) -> np.ndarray:
    # genereate images 
    imgs = cgan.generate_images(input)

    # rescale images (-1, +1) -> (0, 255.0)
    imgs = (imgs[:, :, :] + 1) / 2 * 255.0

    # reshape (num_outputs, 28, 28, 1) -> (num_outputs, 28, 28)
    imgs = imgs.reshape(-1, 28, 28)

    # set dtype to integer
    imgs = imgs.astype('uint8') 

    # return images
    return imgs

def concate_and_save_image(text: str, path: str, num_images: np.ndarray, letter_images: np.ndarray):
    i, j = 0, 0
    h, w = 28, 28

    # create concated image
    concatenated_img = np.zeros((h, w * len(text)), dtype=np.uint8)

    # go through every character
    for ind, c in enumerate(text):
        # if it's a number then add the following number image
        if c in '0123456789':
            concatenated_img[:, ind*w:(ind+1)*w] = num_images[i]
            i += 1

        # otherwise add the following letter image
        else:
            concatenated_img[:, ind*w:(ind+1)*w] = letter_images[j]
            j += 1

    # create an image object
    final_img = Image.fromarray(concatenated_img, mode="L")

    # save the image
    final_img.save(path)

@app.get("/generate/{text}")
def generate_image(text: str = Depends(validate_text)):

    # create the path to the image
    path = os.path.join(imgs_path, f"{text}.jpg")

    # get input labels for the models
    num_labels, letter_labels = get_labels(text)

    # generate images
    if num_labels.size != 0:
        num_images = generate_and_shape_images(cgan_num, num_labels)
    else:
        num_images = []

    if letter_labels.size != 0:
        letter_images = generate_and_shape_images(cgan_letter, letter_labels)
    else:
        letter_images = []
    
    # concate and save the image
    concate_and_save_image(text, path, num_images, letter_images)

    # response 
    return {"text": text, "path": path}

