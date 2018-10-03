import os

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img

base_dir       = '/home/calid/downloads/latte-pics'
train_dir      = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

img_input = layers.Input(shape=(150, 150, 3))

hidden_layers = layers.Conv2D(16, 3, activation='relu')(img_input)
hidden_layers = layers.MaxPooling2D(2)(hidden_layers)

hidden_layers = layers.Conv2D(32, 3, activation='relu')(hidden_layers)
hidden_layers = layers.MaxPooling2D(2)(hidden_layers)

hidden_layers = layers.Conv2D(64, 3, activation='relu')(hidden_layers)
hidden_layers = layers.MaxPooling2D(2)(hidden_layers)

hidden_layers = layers.Flatten()(hidden_layers)
hidden_layers = layers.Dense(512, activation='relu')(hidden_layers)
hidden_layers = layers.Dropout(0.5)(hidden_layers)

output = layers.Dense(1, activation='sigmoid')(hidden_layers)

model = Model(img_input, output)

model.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(lr=0.001),
        metrics=['acc'])

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

history = model.fit_generator(
        train_generator,
        steps_per_epoch=50,
        epochs=15,
        validation_data=validation_generator,
        validation_steps=20,
        verbose=2)

