import numpy as np
import matplotlib.pyplot as plt
from src.PSO import *

class PSO_algorithm:
    def __init__(self, number_of_particles, number_of_colors, image, c1, c2, weight, max_iteration):
        self.number_of_particles = number_of_particles
        self.number_of_colors = number_of_colors
        self.part_population = initial_particle_populus(self.number_of_particles, self.number_of_colors)
        self.image = image
        self.c1 = c1
        self.c2 = c2
        self.weight = weight
        self.max_iteration = max_iteration

    def quantizaton(self):
        iteration = 0
        personal_best = self.part_population.copy()
        global_best_position = personal_best[0].copy()
        global_best_fitness = evalue_fitness(global_best_position, self.image)
        particles = self.part_population.copy()
        while iteration < self.max_iteration+1:
            for i in range(self.number_of_particles):
                fitnes = evalue_fitness(self.part_population[i], self.image)
                # if fitnes < evalue_fitness(personal_best[i], self.image):
                #     print(True)

                if fitnes < global_best_fitness:
                    global_best_position = self.part_population[i].copy()
                    global_best_fitness = fitnes

                new_position = update_velocity_position(self.part_population[i], personal_best[i], global_best_position, self.weight, self.c1, self.c2)
                particles[i] = new_position

            iteration+=1
            return global_best_position


if __name__ == '__main__':
    img = np.random.randint(0, 255, (100, 50, 3), dtype=np.int64)
    # a = plt.imread("./Lenna_(test_image).png")
    # plt.imshow(a)
    # plt.show()
    pso = PSO_algorithm(10, 8, img, 1.2, 1.1, 0.5,50)
    print(pso.quantizaton())