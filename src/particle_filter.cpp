/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <limits>
#include <cassert>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = 500;

  default_random_engine eng;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; i++) {
    struct Particle particle;
    particle.id = i;
    particle.x = dist_x(eng);
    particle.y = dist_y(eng);
    particle.theta = dist_theta(eng);
    particle.weight = 1.0;

    particles.push_back(particle);
  }

  weights.resize(num_particles);

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
  default_random_engine eng;
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++) {
    double new_theta = particles[i].theta + yaw_rate * delta_t + dist_theta(eng);
    if (yaw_rate < 1.0e-10) {
      particles[i].x += velocity * delta_t * cos(particles[i].theta) + dist_x(eng);
      particles[i].y += velocity * delta_t * sin(particles[i].theta) + dist_y(eng);
    }
    else {
      particles[i].x += velocity / yaw_rate * (sin(new_theta) - sin(particles[i].theta)) + dist_x(eng);
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(new_theta)) + dist_y(eng);
    }
    particles[i].theta = new_theta;
  }
}

static double multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs, double mu_x, double mu_y) {
  // calculate normalization term
  double gauss_norm;
  gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

  // calculate exponent
  double exponent;
  exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
             + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));

  // calculate weight using normalization terms and exponent
  double weight;
  weight = gauss_norm * exp(-exponent);

  return weight;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html
  for (int i = 0; i < num_particles; i++) {
    particles[i].weight = 1.0;
    particles[i].associations.clear();
    particles[i].sense_x.clear();
    particles[i].sense_y.clear();
    for (unsigned int o = 0; o < observations.size(); o++) {
      // Convert osbservation coordinates from vehicles' coordinates to global maps' coordinates
      double x_map = particles[i].x + (cos(particles[i].theta) * observations[o].x) -
      (sin(particles[i].theta) * observations[o].y);
      double y_map = particles[i].y + (sin(particles[i].theta) * observations[o].x) +
      (cos(particles[i].theta) * observations[o].y);

      // Find closest landmark
      double min_dist = std::numeric_limits<double>::infinity();
      int predicted_landmark_id = -1;
      int predicted_landmark_index = -1;
      for (unsigned int l = 0; l < map_landmarks.landmark_list.size(); l++) {
        double diff_x = x_map - map_landmarks.landmark_list[l].x_f;
        double diff_y = y_map - map_landmarks.landmark_list[l].y_f;
        double dist = sqrt(diff_x * diff_x + diff_y * diff_y);
        if (dist < sensor_range && dist < min_dist) {
          min_dist = dist;
          predicted_landmark_id = map_landmarks.landmark_list[l].id_i;
          predicted_landmark_index = l;
        }
      }
      if (predicted_landmark_id != -1) {
        particles[i].sense_x.push_back(x_map);
        particles[i].sense_y.push_back(y_map);
        particles[i].associations.push_back(predicted_landmark_id);

        // Calculate particle weight according to current observation
        particles[i].weight *= multiv_prob(std_landmark[0], std_landmark[1], x_map, y_map,
                                           map_landmarks.landmark_list[predicted_landmark_index].x_f,
                                           map_landmarks.landmark_list[predicted_landmark_index].y_f);
      }
      else {
        particles[i].weight = 0.0;
      }
    }
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  default_random_engine eng;
  discrete_distribution<> d(weights.begin(), weights.end());

  vector<Particle> new_particles;
  for (int i = 0; i < num_particles; i++) {
    Particle p = particles[d(eng)];
    new_particles.push_back(p);
  }
  particles = new_particles;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
