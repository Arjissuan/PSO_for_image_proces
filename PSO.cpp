#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class PSOAlgorithm {
public:
    int number_of_particles;
    int number_of_colors;
    float c1, c2, weight;
    int max_iteration;

    PSOAlgorithm(int num_particles, int num_colors, float c1, float c2, float weight, int max_iter)
        : number_of_particles(num_particles), number_of_colors(num_colors), c1(c1), c2(c2), weight(weight), max_iteration(max_iter) {}

    vector<vector<Vec3b>> initialParticlePopulus(int num_particles, int num_colors) {
        vector<vector<Vec3b>> population;
        for (int i = 0; i < num_particles; ++i) {
            vector<Vec3b> particle;
            for (int j = 0; j < num_colors; ++j) {
                particle.push_back(Vec3b(rand() % 256, rand() % 256, rand() % 256));
            }
            population.push_back(particle);
        }
        return population;
    }

    float evaluateFitness(const vector<Vec3b>& particle, const Mat& image) {
        float fitness = 0.0;
        for (const auto& color : particle) {
            Mat distances;
            cv::pow(image - Scalar(color[0], color[1], color[2]), 2, distances);
            reduce(distances, distances, 2, REDUCE_SUM);
            double min_distance;
            cv::minMaxLoc(distances, &min_distance);
            fitness += min_distance;
        }
        return fitness;
    }

    vector<Vec3b> updateVelocityPosition(const vector<Vec3b>& particle,
                                         const vector<Vec3b>& best_particle,
                                         const vector<Vec3b>& best_global_particle,
                                         float w, float c1, float c2) {
        vector<Vec3b> velocity, new_particle;
        for (size_t i = 0; i < particle.size(); ++i) {
            Vec3b rand1(rand() % 256, rand() % 256, rand() % 256);
            Vec3b rand2(rand() % 256, rand() % 256, rand() % 256);
            velocity.push_back(w * Vec3b(rand() % 256, rand() % 256, rand() % 256) + c1 * rand1 * (best_particle[i] - particle[i]) + c2 * rand2 * (best_global_particle[i] - particle[i]));
            new_particle.push_back(particle[i] + velocity[i]);
        }
        return new_particle;
    }

    Mat quantization(const Mat& image) {
        int iteration = 0;
        auto initial_particle_population = initialParticlePopulus(number_of_particles, number_of_colors);
        auto particles = initial_particle_population;
        auto personal_best = initial_particle_population;
        auto global_best_position = personal_best[0];
        auto global_best_fitness = evaluateFitness(global_best_position, image);

        while (iteration < max_iteration + 1) {
            cout << iteration << endl;
            for (size_t i = 0; i < number_of_particles; ++i) {
                float fitness = evaluateFitness(particles[i], image);

                if (fitness < evaluateFitness(personal_best[i], image)) {
                    personal_best[i] = particles[i];
                }

                if (fitness < global_best_fitness) {
                    global_best_position = particles[i];
                    global_best_fitness = fitness;
                }

                auto new_position = updateVelocityPosition(particles[i], personal_best[i], global_best_position, weight, c1, c2);
                particles[i] = new_position;
            }
            iteration += 1;
        }

        Mat pixels = image.reshape(1, image.rows * image.cols);
        Mat pixels_float;
        pixels.convertTo(pixels_float, CV_32F);
        Mat labels, quantized_image;
        kmeans(pixels_float, number_of_colors, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 1.0), 1, KMEANS_RANDOM_CENTERS);
        Mat quant_centroids;
        kmeans(image.reshape(1, image.rows * image.cols), number_of_colors, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 1.0), 1, KMEANS_RANDOM_CENTERS, quant_centroids);

        quantized_image = quant_centroids.row(labels).reshape(3, image.rows);
        quantized_image.convertTo(quantized_image, CV_8U);

        return quantized_image;
    }
};

int main() {
    Mat image_lenna = imread("test_images/Lenna_(test_image).png");
    Mat image_goldhill = imread("test_images/goldhill.bmp");
    PSOAlgorithm pso(5, 10, 1.2, 1.1, 0.6, 30);

    Mat quantized_image_lenna = pso.quantization(image_lenna);
    Mat quantized_image_goldhill = pso.quantization(image_goldhill);

    imshow("Original Image - Lenna", image_lenna);
    imshow("Quantized Image - Lenna", quantized_image_lenna);
    waitKey(0);

    imshow("Original Image - Goldhill", image_goldhill);
    imshow("Quantized Image - Goldhill", quantized_image_goldhill);
    waitKey(0);

    return 0;
}
