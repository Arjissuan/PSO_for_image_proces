import numpy as np

def initial_particle_populus(num_par, num_col):
    population = []
    for i in range(num_par):
        particle = np.random.randint(0, 256, size=(num_col, 3), dtype=np.int32)
        population.append(particle)
    return population

def evalue_fitness(particle, image):
    fitness = 0
    for color in particle:
        distances = np.linalg.norm(image - color, axis=2)
        fitness += np.min(distances)
    return fitness

def update_velocity_position(particle, best_particle, best_global_particle, w, c1, c2):
    velocity = np.random.uniform(0, 1, size=particle.shape)
    r1 = np.random.uniform(0, 1, size=particle.shape)
    r2 = np.random.uniform(0, 1, size=particle.shape)
    new_velocity = w * velocity + c1 * r1 * (best_particle - particle) + c2 * r2 * (best_global_particle - particle)
    new_particle = particle + new_velocity
    return new_particle

