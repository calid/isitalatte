import os
import tensorflow.keras as tfk

from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tfk.models.load_model('isitalatte.h5')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        'latte-pics/test',
        target_size=(150, 150),
        batch_size=15,
        class_mode='binary')

metrics = model.evaluate_generator(test_generator, verbose=1)
for idx, name in enumerate(model.metrics_names):
    print("%s: %0.2f" % (name, metrics[idx]))
