import os

import tensorflow.keras as tfk
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image

import numpy
import argparse

# suppress a warning about unused cpu instructions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description='Is the image a latte?')
parser.add_argument('image', type=str,
        help='An image that may or may not contain a latte')

args = parser.parse_args()

model = tfk.models.load_model('isitalatte.h5')

test_image = image.load_img(args.image, target_size=(150,150))
test_image = image.img_to_array(test_image)
test_image /= 255. # rescale
test_image = numpy.expand_dims(test_image, axis=0) # reshape for single test
result = model.predict(test_image)

# Class 0 = Latte
# Class 1 = Not-Latte
# (order the fit generator found the example directories during training)

# result is the probability the image belongs to class 1, not-latte
not_latte_prob = result[0][0]
latte_prob     = 1.0 - not_latte_prob

if latte_prob > 0.7:
    print("It's a latte!")
elif not_latte_prob > 0.7:
    print("Not a latte")
else:
    print("Not sure if latte?")

print()
print("Latte: %0.3f%%" % (latte_prob*100))
print("Not Latte: %0.3f%%" % (not_latte_prob*100))
