import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images/255, testing_images/255

class_names = ['Plane','Car','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

datagen = ImageDataGenerator(
  rotation_range = 10,
  width_shift_range = 0.05,
  height_shift_range = 0.05,
  horizontal_flip = True
)

datagen.fit(training_images)


#
# training_images = training_images[:20000]
# training_labels = training_labels[:20000]
# testing_images = testing_images[:4000]
# testing_labels = testing_labels[:4000]


######## model training ########

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3,3), activation = 'relu', padding='same', input_shape=(32,32,3)))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(32,(3,3), activation = 'relu', padding='same'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Dropout(0.2))

# model.add(layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D(2,2))
# model.add(layers.Dropout(0.3))

# model.add(layers.Conv2D(128, (3,3), padding = 'same', activation = 'relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(128, (3,3), padding = 'same', activation = 'relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.GlobalAveragePooling2D())
# model.add(layers.Dropout(0.4))

# model.add(layers.Flatten())
# model.add(layers.Dense(10, activation='softmax'))



# model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# early_stop = EarlyStopping(monitor='val_accuracy', patience = 5, restore_best_weights=True)

# model.fit(datagen.flow(training_images, training_labels, batch_size=64), epochs=30, callbacks=[early_stop], validation_data=(testing_images,testing_labels))

# loss, accuracy = model.evaluate(testing_images, testing_labels)
# print(f"Loss: {loss}")
# print(f"Accuracy: {accuracy}")

# model.save('image_classifier.keras')


model = models.load_model('image_classifier.keras')

img = cv.imread('deer.png')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f"Prediction is {class_names[index]}")

plt.show()