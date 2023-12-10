# KepKreator

## Team members:

| Name | Neptun |
| --- | --- |
| Aszódi Zsombor | XJ43M0  |
| Babos Dávid | Q1CGY7 |
| Kovács Gergely | JWV9WR |

## The project

The plan is to generate images using Conditional Generative Adversarial Nets (cGANs). As other cGANs, the network will consist of two models, a Generator and a Discriminator. The input to the Generator will be a noise vector (also called as latent vector) and the output will be a generated image. The input of the Discriminator will be an image, either a 'real' image from the training dataset or a 'fake' image, created by the Generator. The Discriminator's role is to decide whether the input image is real or fake. The goal is to create a Conditional GAN, hence the input (to both the Generator and the Discriminator) is concatenated with a label. The desired outcome is to generate hand-written numbers and characters. At first trying to create one-digit numbers, than upgrade the model to create multi-digit ones and letters. The team used the MNIST and the EMNIST datasets, to train the model. At the end there is an interface, where the user can give an input number/letter, which the model than generates.

## Model training & data evaluation

The assembled cGAN model, which works on the basic MNIST database can be found in the *MNIST_GAN.ipynb* file. After the defining of the model you can find the training in the same file under the *Create models and train* subtitle. For evaluation, we saved the model after various epochs and we plotted some generated image examples after the training of the model. Thus you can see, how the model learns after each epoch and can evaluate the created images throughout the training process. The data evaluation can be found under the *Load model* subtitle. The hyperparameter optimization is under the subtitle *Hypermodel for hyperparameter optimization*.

The layout is similar for the EMNIST based cGAN model, because it’s basically the same model, but it had been tuned to the EMNIST database. So the training and the data evaluation is the same as described previously. The main difference is, that there were two data preprocessing created. The first one is for the characters that were used in the training of the model. It can be found under the *Preprocess Letters dataset* subtitle. The other one preprocesses the whole EMNIST database, and it had been created for possible future trainings. It is found under the *Preprocess ByClass dataset* subtitle.

## Installation

To try out the pre-trained models with a web interface first of all you have to clone the project

```bash
git clone https://github.com/FokaKefir/KepKreator.git
```

Move to the `app` library 

```bash
cd app/
```

Install the dependencies

```bash
pip install -r requirements.txt
```

Start the API service 

If you are using git-bash or Linux just run the `start.sh` script

```bash
./start.sh
```

Otherwise just type in the following command in the shell

```bash
uvicorn generate:app --reload
```

Wait until you see the `Ready to go!` message, then just open `web/index.html` in your browser

To stop the service just press `Ctrl+C`

Feel free to generate images like this:

![docs/example.jpg]()

## Files in repository

| Source | Description |
| --- | --- |
| MNIST_preprocess.ipynb | Imports the MNIST database |
| MNIST2_preprocess.ipynb | Imports the MNIST database and makes 2 two digit numbers  |
| EMNIST_preprocess.ipynb | Imports the EMNIST database |
| MNIST_GAN.ipynb | The cGAN model, which works on the basic MNIST database |
| EMNIST_GAN.ipynb | The cGAN model, which works on the EMNIST database |
| app repository | Contains all the files for the model interface |
| validation_cnn.hdf5 | A small model with accuracy around 99% for inception score calculation |
| MNIST_val_model.ipynb | Used for the training of the validation_cnn |

## Citations

"THE MNIST DATABASE of handwritten digits". Yann LeCun, Courant Institute, NYU Corinna Cortes, Google Labs, New York Christopher J.C. Burges, Microsoft Research, Redmond. Retrieved from: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters. Retrieved from [http://arxiv.org/abs/1702.05373](http://arxiv.org/abs/1702.05373)