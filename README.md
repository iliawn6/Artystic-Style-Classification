# 🖼️ Who Painted This? Identifying Artists with Deep Learning (ResNet50)

This project explores the question:

> _Can a neural network develop a sense of artistic recognition?_

We train a convolutional neural network to classify paintings by famous artists.  
The challenge: given an unknown painting, can the model predict **who the painter is**?
<br><br>

## 🎯 Objectives

- Build a deep learning model that classifies paintings by artist.  
- Explore **transfer learning** and **data augmentation** to improve performance.  
- Compare different training strategies using the **ResNet50** architecture.  

The dataset contains several hundred paintings from multiple world-renowned artists.
<br><br>


## 🧠 Model Architecture: ResNet50

We use **ResNet50**, a widely adopted CNN architecture introduced by Microsoft researchers.  

- ResNet solves the **vanishing gradient problem** with **Residual Blocks**.  
- Each block includes a **skip connection** that directly passes the input to the output.  
- This allows deep networks to be trained effectively, even with 50+ layers.  

In Keras, ResNet50 is easily accessible via:

```python
from keras.applications import ResNet50
```
<br><br>

## 🔁 Transfer Learning Strategies

We experimented with different strategies for applying transfer learning with ResNet50:

1. **Training from scratch**  
   - Model initialized with random weights.  
   - Low accuracy, struggled due to limited dataset size.  

2. **Fine-tuning all layers (with pre-trained weights)**  
   - Initialized with ImageNet weights, trained the entire network.  
   - Better accuracy, but slower training and risk of overfitting.  

3. **Freezing early layers, training only the final layers**  
   - Retained general features (edges, textures) from pre-trained weights.  
   - Adapted only the higher-level layers for painter classification.  
   - **Best-performing approach** for our dataset.
<br><br>

## 📊 Results

| Strategy                                   | Accuracy |
|--------------------------------------------|----------|
| Training from scratch                      | Low      |
| Fine-tuning all layers (pre-trained)       | Medium   |
| **Freezing early layers + training final layers** | **~98%** |

Freezing the early layers and training only the final ones achieved the highest accuracy (~98%).  
This confirms that transfer learning with selective fine-tuning could be the most effective method for smaller datasets like ours.
