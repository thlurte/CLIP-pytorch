# CLIP

<p align="center">
  <img src="assets/a.png" alt="">
</p>

CLIP helps us to learn representations of images and text in a shared embedding space.

## Dataset

<p align="center">
  <img src="assets/b.png" alt="">
</p>

In this case, `Flickr 8k Dataset` has been used to train and validate the model. The dataset comprises of 8,000 images that are each paired with five different captions which provide clear descriptions of the salient entities and events. Since we are training the model both on text and image, the dataset should to return both images and texts.

DistilBERT tokenizor from huggingface will be used to tokenize the sentences.