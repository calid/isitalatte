import os

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3

import matplotlib
matplotlib.use("agg")

import matplotlib.pyplot as plt
import matplotlib.image  as mpimg

base_dir       = 'latte-pics'
train_dir      = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

pre_trained_model = InceptionV3(
        input_shape=(150, 150, 3),
        include_top=False,
        weights=None)

pre_trained_model.load_weights(
    'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer  = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output

hidden_layers = layers.Flatten()(last_output)
hidden_layers = layers.Dense(1024, activation='relu')(hidden_layers)
hidden_layers = layers.Dropout(0.2)(hidden_layers)
hidden_layers = layers.Dense(1, activation='sigmoid')(hidden_layers)

model = Model(pre_trained_model.input, hidden_layers)
model.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(lr=0.001),
        metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

validation_datagen  = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=15,
        class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=15,
        class_mode='binary')

print(f'Class ids: {train_generator.class_indices}')

history = model.fit_generator(
        train_generator,
        steps_per_epoch=50,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=20,
        verbose=2)

train_acc = history.history['acc']
val_acc   = history.history['val_acc']

train_loss = history.history['loss']
val_loss   = history.history['val_loss']

epochs = range(len(train_acc))

plt.figure(figsize=(12, 12))

ax = plt.subplot(2, 1, 1)
ax.set_title('Training and Validation Accuracy')
ax.set_xlabel("Epochs")
ax.set_ylabel("Accuracy")
ax.plot(epochs, train_acc, label="training")
ax.plot(epochs, val_acc, label="validatoin")
ax.legend()

ax = plt.subplot(2, 1, 2)
ax.set_title('Training and Validation Loss')
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.plot(epochs, train_loss, label="training")
ax.plot(epochs, val_loss, label="validatoin")
ax.legend()

plt.savefig('accuracy_and_loss.png')

tf.keras.models.save_model(model, 'isitalatte.h5')
