# CS236DefaultProject

This is the default project for [CS236: Deep Generative Models](https://deepgenerativemodels.github.io/). You'll be implementing text-conditioned image generation.

## Getting Started

First, install the required python libraries as follows. You'll need Python 3.6 or greater.

```
pip -r requirements.txt
```

Next, download the ImageNet32 and class names to the `datasets` directory by executing:

```
bash download_data.sh
```

## Todos

We have provided a dataloader for the Tier-1 dataset in `data.py`. It outputs pairs of images and class names.

In `models/starter_models.py` you will find the **BERT encoder**  that you will use to extract features from text. It is a pretrained model and should not be fine-tuned.

You will also find the interface `CaptionConditionedGenerativeModel` that your models have to implement. Its main method is the `forward` method that takes as input a batch of images `x`, a batch of text embeddings `h` and outputs the loss of your model. Your model should also implement a `sample` method and if it is a likelihood model, a `likelihood` method.

A function to sample images and display them with the conditioned captions is given in `utils.py`.