import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt
from src.PSO import *

class PSO_algorithm:
    def __init__(self, number_of_particles, number_of_colors, c1, c2, weight, max_iteration):
        self.number_of_particles = number_of_particles
        self.number_of_colors = number_of_colors
        self.initial_particle_populus = initial_particle_populus(self.number_of_particles, self.number_of_colors)
        self.c1 = c1
        self.c2 = c2
        self.weight = weight
        self.max_iteration = max_iteration

    def quantizaton(self, image):
        iteration = 0
        particles = self.initial_particle_populus.copy()
        personal_best = self.initial_particle_populus.copy()
        global_best_position = personal_best[0].copy()
        global_best_fitness = evalue_fitness(global_best_position, image)
        while iteration < self.max_iteration+1:
            for i in range(self.number_of_particles):
                fitnes = evalue_fitness(particles[i], image)
                # if fitnes < evalue_fitness(personal_best[i], self.image):
                #     print(True)

                if fitnes < global_best_fitness:
                    global_best_position = particles[i].copy()
                    global_best_fitness = fitnes

                new_position = update_velocity_position(particles[i], personal_best[i], global_best_position, self.weight, self.c1, self.c2)
                particles[i] = new_position


            iteration+=1

        pixels = image.reshape((-1,3))
        kmeans = KMeans(n_clusters=len(global_best_position), init=global_best_position, n_init=1, random_state=42)
        kmeans.fit(pixels)
        labels = kmeans.labels_
        quant_centroids = kmeans.cluster_centers_
        quantized_image = quant_centroids[labels].reshape(image.shape)
        quantized_image = np.uint8(quantized_image)

        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.title('Original Image')
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(1,2,2)
        plt.title('Quantized Image')
        plt.imshow(quantized_image)
        plt.axis('off')

        plt.show()
        return quantized_image


if __name__ == '__main__':
    image_goldhill = np.array(Image.open('test_images/goldhill.bmp'))
    image_lenna = np.array(Image.open("test_images/Lenna_(test_image).png"))
    # plt.imshow(a)
    # plt.show()
    pso = PSO_algorithm(10, 8, 1.2, 1.1, 0.5,50)
    pso.quantizaton(image_lenna)
    pso.quantizaton(image_goldhill)

