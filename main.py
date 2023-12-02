from PIL import Image
import numpy as np
from src.PSO import PSO_algorithm


if __name__ == '__main__':
    image_goldhill = np.array(Image.open('test_images/goldhill.bmp'))
    image_lenna = np.array(Image.open("test_images/Lenna_(test_image).png"))
    # plt.imshow(a)
    # plt.show()
    pso = PSO_algorithm(10, 10, 1.2, 1.1, 0.6,50)
    pso.quantizaton(image_lenna)
    pso.quantizaton(image_goldhill)

