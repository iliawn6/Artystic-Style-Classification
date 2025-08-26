# 🖼️ Who Painted This? Identifying Artists with Deep Learning (ResNet50)

This project explores the question:

> _Can a neural network develop a sense of artistic recognition?_

We train a convolutional neural network to classify paintings by famous artists.  
The challenge: given an unknown painting, can the model predict **who the painter is**?

---

## 🎯 Objectives

- Build a deep learning model that classifies paintings by artist.  
- Explore **transfer learning** and **data augmentation** to improve performance.  
- Compare different training strategies using the **ResNet50** architecture.  

The dataset contains several hundred paintings from multiple world-renowned artists.

---

## 🧠 Model Architecture: ResNet50

We use **ResNet50**, a widely adopted CNN architecture introduced by Microsoft researchers.  

- ResNet solves the **vanishing gradient problem** with **Residual Blocks**.  
- Each block includes a **skip connection** that directly passes the input to the output.  
- This allows deep networks to be trained effectively, even with 50+ layers.  

In Keras, ResNet50 is easily accessible via:

```python
from keras.applications import ResNet50
