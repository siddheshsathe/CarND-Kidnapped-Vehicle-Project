/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
static std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles

  std::normal_distribution<double> distX(x, std[0]);
  std::normal_distribution<double> distY(y, std[1]);
  std::normal_distribution<double> distTheta(theta, std[2]);

  for(int i=0; i<num_particles; i++){
    Particle p;
    p.id = i;
    p.x = distX(gen);
    p.y =  distY(gen);
    p.theta = distTheta(gen);
    p.weight = 1.0;
    particles.push_back(p);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
    for(auto &p: particles){
        double newX, newY, newTheta;
        if(abs(yaw_rate) > 1e-05){
            newTheta = p.theta + yaw_rate * delta_t;
            newX = p.x + velocity/yaw_rate * (sin(newTheta) - sin(p.theta));
            newY = p.y + velocity/yaw_rate * (cos(p.theta) - cos(newTheta));
        }
        else{ // If yaw rate is less than 1e-05, then just update the theta, but don't use it for newX and newY
            newTheta = p.theta + yaw_rate * delta_t;
            newX = p.x + velocity * delta_t * cos(p.theta);
            newY = p.y + velocity * delta_t * sin(p.theta);
        }

        std::normal_distribution<double> distX(newX, std_pos[0]);
        p.x = distX(gen);

        std::normal_distribution<double> distY(newY, std_pos[1]);
        p.y = distY(gen);

        std::normal_distribution<double> distTheta(newTheta, std_pos[2]);
        p.theta = distTheta(gen);
    }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

    for (auto& observation : observations) {
        double min_dist = std::numeric_limits<double>::max();

        for (const auto& pred_obs : predicted) {
            double d = dist(observation.x, observation.y, pred_obs.x, pred_obs.y);
            if (d < min_dist) {
                observation.id   = pred_obs.id;
                min_dist = d;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
    for(auto &p: particles){
        // for(size_t i=0; i < particles.size(); i++){
        // Nearest landmarks in sensor_range
        vector<LandmarkObs> nearest_landmarks;
        for(auto &map_landmark: map_landmarks.landmark_list){
            double distance = dist(p.x, p.y, map_landmark.x_f, map_landmark.y_f);
            if(distance < sensor_range){
                LandmarkObs nearL;
                nearL.id = map_landmark.id_i;
                nearL.x = map_landmark.x_f;
                nearL.y = map_landmark.y_f;
                nearest_landmarks.push_back(nearL);
            }
        }

        vector<LandmarkObs> observed_landmarks;
        for(auto &obs: observations){
            LandmarkObs translatedLandmark;
            translatedLandmark.x = p.x + cos(p.theta) * obs.x - sin(p.theta) * obs.y;
            translatedLandmark.y = p.y + sin(p.theta) * obs.x + cos(p.theta) * obs.y;

            observed_landmarks.push_back(translatedLandmark);
        }

        // Doing data association to find which landmarks the observations correspond
        dataAssociation(nearest_landmarks, observed_landmarks);

        double muX, muY;
        double likelihood = 1.0;
        for(const auto &obs: observed_landmarks){
            for(const auto &nearestLandmark: nearest_landmarks){
                if(obs.id == nearestLandmark.id){
                    muX = nearestLandmark.x;
                    muY = nearestLandmark.y;
                    break;
                }
            }
            double normFactor = 2 * M_PI * std_landmark[0] * std_landmark[1];
            double probability = pow(M_E, -(pow(obs.x - muX, 2) / (2 * pow(std_landmark[0], 2)) + pow(obs.y - muY, 2) / (2 * pow(std_landmark[1], 2))));

            likelihood *= probability / normFactor;

            p.weight = likelihood;
        }
    }

        double normFactor = 0.0;
        for(auto &p: particles)
            normFactor += p.weight;

        for(auto &p: particles)
            p.weight /= (normFactor + M_E);

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
       vector<double> particle_weights;
    for (const auto& particle : particles)
        particle_weights.push_back(particle.weight);

    std::discrete_distribution<int> weighted_distribution(particle_weights.begin(), particle_weights.end());

    vector<Particle> resampled_particles;
    for (int i = 0; i < num_particles; ++i) {
        int k = weighted_distribution(gen);
        resampled_particles.push_back(particles[k]);
    }

    particles = resampled_particles;

    // Reset weights for all particles
    for (auto& particle : particles)
        particle.weight = 1.0;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}