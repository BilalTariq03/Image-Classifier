# Image Classifier using CNN (TensorFlow/Keras)

This is a simple image classification project built using Convolutional Neural Networks (CNNs) with TensorFlow and Keras. The model is trained on the CIFAR-10 dataset to classify images into 10 categories.

---

## Dataset

We used the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), which contains 60,000 32x32 color images in 10 classes, with 6,000 images per class.

---

## Features

- Achieves **~85% accuracy** on the test set
- Uses **Convolutional Neural Networks (CNNs)** for image classification
- Includes **data augmentation** for better generalization
- Implements **Batch Normalization**, **Dropout**, and **Early Stopping**
- Saves and loads trained model for future inference

## Classes

- Airplane 
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Technology Used 
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib

## Dataset
- CIFAR-10, a collection of 60,000 32x32 color images in 10 classes, with 6,000 images per class.
- 50,000 training images and 10,000 test images.
- Automatically downloaded using `tensorflow.keras.datasets`.

## Model Architecture

```python
Conv2D(32) → BatchNorm → Conv2D(32) → BatchNorm → MaxPooling → Dropout
→ Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPooling → Dropout
→ Conv2D(128) → BatchNorm → Conv2D(128) → BatchNorm → GlobalAveragePooling
→ Dropout → Dense(10, softmax)
```

## Training Strategy
- Data Augmentation: rotation, shifts, horizontal flips
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Early Stopping: Stops training when validation accuracy stops improving
- Batch Size: 64
- Epochs: Up to 30 (early stopping enabled)

## How to use
1. Clone reopsitory
   ```bash
    git clone https://github.com/YourUsername/ImageClassifier.git
   cd ImageClassifier
   ```

2. Install dependencies
   ```bash
   pip install tensorflow numpy opencv-python matplotlib
    ```
3. Add image to images folder
4. Run the script
    - To train and save the model (if you want to retrain):
      
    Uncomment the training block in ImageClassification.py, then run:
    ```bash
    python ImageClassification.py
    ```
    - To load the model and classify an image:
      
    Make sure model is saved as image_classifier.keras and an image path is set in the script:
    ```bash
    python ImageClassification.py
    ```
5. View the result
The model will print the predicted class (e.g., Prediction is Deer)

And display the input image using matplotlib

## Future Improvements
- Implement transfer learning with pretrained models like ResNet or MobileNet
- Deploy as a web app using Flask or Streamlit
- Export to ONNX for cross-platform usage

## License
This project is licensed under the MIT License.

