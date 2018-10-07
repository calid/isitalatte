import tensorflow as tf
import numpy as np
from random import randint

IMG_WIDTH = 250
IMG_HEIGHT = 250

class NormalizeImageTest(tf.test.TestCase):

    def testNormalizeImage(self):
        normalize_img_module = tf.load_op_library('./normalize_image.so')

        with self.test_session():
            (origImage, normalizedImage) = self.genTestData()

            # sanity check our input/expected output
            self.assertAllEqual(origImage.shape, (IMG_HEIGHT, IMG_WIDTH, 3))
            self.assertAllEqual(normalizedImage.shape, (IMG_HEIGHT, IMG_WIDTH, 3))

            result = normalize_img_module.normalize_image(origImage).eval()

            self.assertAllEqual(result.shape, normalizedImage.shape)
            np.testing.assert_almost_equal(result, normalizedImage)
            self.assertDTypeEqual(result, 'float32')

    def genTestData(self):
        # generate fake image data of shape (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
        # generate both fake 'original' image data and the expected
        # normalized data (for normalization we are rescaling 3 channel rgb
        # data from 0-255 to 0-1
        origImage = []
        normalizedImage = []

        for i in range(IMG_HEIGHT):
            origImage.append([])
            normalizedImage.append([])

            for j in range(IMG_WIDTH):
                (nextRgb, normalizedRgb) = self.randomRgb()
                origImage[i].append(nextRgb)
                normalizedImage[i].append(normalizedRgb)

        origImage = np.array(origImage)
        normalizedImage = np.array(normalizedImage)

        return (origImage, normalizedImage)

    def randomRgb(self):
        # generate both "original rgb" values
        # and the values we expect after normalization
        origRgb = []
        normalizedRgb = []

        for i in range(3):
            nextPixel = randint(0,255)
            normalizedPixel = nextPixel / 255.

            origRgb.append(nextPixel)
            normalizedRgb.append(normalizedPixel)

        return (origRgb, normalizedRgb)


if __name__ == "__main__":
    tf.test.main()
