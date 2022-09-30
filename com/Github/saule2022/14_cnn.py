import pathlib
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Rescaling, Dense, Conv2D, Flatten, Dropout, MaxPooling2D

data_path = pathlib.Path(r'C:\DATA\various_datasets\dogs-vs-cats\train')

train_set = tf.keras.utils.image_dataset_from_directory(
    data_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(64, 64),
    batch_size=100)


validation_set = tf.keras.utils.image_dataset_from_directory(
    data_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(64, 64),
    batch_size=100)


class_names = train_set.class_names
print(class_names)

# plt.figure(figsize=(10, 10))
# for images, labels in train_set.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")


model = Sequential([
    Rescaling(1. / 255, input_shape=(64, 64, 3)),
    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.1),
    Flatten(),
    Dense(1, activation='sigmoid')])

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()


history = model.fit(
    train_data[0],
    validation_data=validation_data[0],
    epochs=3)