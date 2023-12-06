import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from typing import List

class PSO_algorithm:
    def __init__(self, number_of_particles:int, number_of_colors:int, c1:float, c2:float, weight:float, max_iteration:int):
        self.number_of_particles = number_of_particles
        self.number_of_colors = number_of_colors
        self.c1 = c1
        self.c2 = c2
        self.weight = weight
        self.max_iteration = max_iteration

    def initial_particle_populus(self, num_par: int, num_col: int) -> list:
        population = []
        for i in range(num_par):
            particle = np.random.randint(0, 256, size=(num_col, 3), dtype=np.int32)
            population.append(particle)
        return population

    def evalue_fitness(self, particle: np.ndarray, image: np.ndarray) -> float:
        fitness = 0.0
        for color in particle:
            distances = np.linalg.norm(image - color, axis=2)
            fitness += np.min(distances)
        return fitness

    def update_velocity_position(self, particle: np.ndarray,
                                 best_particle: np.ndarray,
                                 best_global_particle:
                                 np.ndarray,
                                 w: float, c1: float, c2: float) -> np.ndarray:
        velocity = np.random.uniform(0, 1, size=particle.shape)
        r1 = np.random.uniform(0, 1, size=particle.shape)
        r2 = np.random.uniform(0, 1, size=particle.shape)
        new_velocity = w * velocity + c1 * r1 * (best_particle - particle) + c2 * r2 * (best_global_particle - particle)
        new_particle = particle + new_velocity
        return new_particle

    def quantizaton(self, image: np.ndarray) -> np.ndarray:
        iteration = 0
        initial_particle_population = self.initial_particle_populus(self.number_of_particles, self.number_of_colors)
        particles = initial_particle_population.copy()
        personal_best = initial_particle_population.copy()
        global_best_position = personal_best[0].copy()
        global_best_fitness = self.evalue_fitness(global_best_position, image)
        while iteration < self.max_iteration+1:
            print(iteration)
            for i in range(self.number_of_particles):
                fitnes = self.evalue_fitness(particles[i], image)

                if fitnes < self.evalue_fitness(personal_best[i], image):
                    personal_best[i] = particles[i]

                if fitnes < global_best_fitness:
                    global_best_position = particles[i].copy()
                    global_best_fitness = fitnes

                new_position = self.update_velocity_position(particles[i], personal_best[i], global_best_position, self.weight, self.c1, self.c2)
                particles[i] = new_position
            iteration += 1

        pixels = image.reshape((-1, 3))
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