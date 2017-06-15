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

#include "particle_filter.h"

using namespace std;

// generator for random numbers
default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// set the number of particles to useful
	num_particles = 200;

	// define the normal distributions for the given uncertainties std[std_x, std_y, std_theta]
	// sample from these normal distrubtions like this:
	// sample_x = dist_x(gen);
	normal_distribution<double> nd_x(0, std[0]);
  normal_distribution<double> nd_y(0, std[1]);
  normal_distribution<double> nd_theta(0, std[2]);

	// Initialize all particles to the first position
	for(int i=0; i<num_particles; i++){

		Particle particle;
		// fill out the particle variable with the init values and their respective noise
		particle.id = i;
		particle.x = x + nd_x(gen);
		particle.y = y + nd_y(gen);
		particle.theta = theta + nd_theta(gen);
		particle.weight = 1.;

		// add this new particle to the public particles vector
		particles.push_back(particle);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// define the normal distributions for the given measurement uncertainties std_pos[std_x, std_y, std_theta]
	// sample from these normal distrubtions like this:
	// sample_x = dist_x(gen);
	normal_distribution<double> nd_x(0, std_pos[0]);
	normal_distribution<double> nd_y(0, std_pos[1]);
	normal_distribution<double> nd_theta(0, std_pos[2]);

	// iterate through the elements of the public particles vector
	for(int i=0; i<particles.size(); i++){

		// to speed things up since this is used many times below
		double theta = particles[i].theta;
		double theta_p_theta_dot_dt = yaw_rate*delta_t + theta;

		if(abs(yaw_rate) < 0.00001){

			// add measurements
			particles[i].x += velocity*delta_t*cos(theta);
			particles[i].y += velocity*delta_t*sin(theta);
		} else {

			// add measurements
			particles[i].x += (velocity/yaw_rate)*(sin(theta_p_theta_dot_dt)-sin(theta));
			particles[i].y += (velocity/yaw_rate)*(cos(theta)-cos(theta_p_theta_dot_dt));
			particles[i].theta = theta_p_theta_dot_dt;
		}


		// add noise
		particles[i].x += nd_x(gen);
		particles[i].y += nd_y(gen);
		particles[i].theta += nd_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	for(int i=0; i<observations.size(); i++){

		int new_id = -1;
		double closest_dst = numeric_limits<double>::max();
		LandmarkObs curr_obs = observations[i];

		for(int j=0; j<predicted.size(); j++){

			LandmarkObs iter_pred = predicted[j];

			// use the <iterator> function dist() to get the distance between two grphical points
			double iter_dst = dist(curr_obs.x,curr_obs.y,iter_pred.x,iter_pred.y);

			if(iter_dst<closest_dst){

				closest_dst = iter_dst;
				new_id = predicted[j].id;
			}
		}
		observations[i].id = new_id;
		// observations[i].x = predicted[new_id].x;
		// observations[i].y = predicted[new_id].y;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
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

	for(int i=0; i<num_particles; i++){

		// current particle parameters
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;

		// prediction storage
		vector<LandmarkObs> prediction_list;

		// iterate and check for predictions
		for(int j=0; j<map_landmarks.landmark_list.size(); j++){

			// landmark parameters
			int lmark_id = map_landmarks.landmark_list[j].id_i;
			double lmark_x = map_landmarks.landmark_list[j].x_f;
			double lmark_y = map_landmarks.landmark_list[j].y_f;

			if(dist(p_x, p_y, lmark_x, lmark_y) <= sensor_range){

				prediction_list.push_back(LandmarkObs{lmark_id, lmark_x, lmark_y});
			}
		}

		// transform the observations to the map coordinate system
		vector<LandmarkObs> observations_map;
		for(int j=0; j<observations.size(); j++){

      double map_x = p_x+cos(p_theta)*observations[j].x-sin(p_theta)*observations[j].y;
      double map_y = p_y+sin(p_theta)*observations[j].x+cos(p_theta)*observations[j].y;
      observations_map.push_back(LandmarkObs{observations[j].id, map_x, map_y});
		}

		// go for the data association
		dataAssociation(prediction_list, observations_map);

		// get the new weights
		// re initializing weights back to 1.
		particles[i].weight = 1.;

		// placeholders for the upcoming loop
		int da_id;
		double obs_x, obs_y, pred_x, pred_y;

		// iterate through the observations_map
		for(int j=0; j<observations_map.size(); j++){

			da_id = observations_map[j].id;
			obs_x = observations_map[j].x;
			obs_y = observations_map[j].y;

			for(int k=0; k<prediction_list.size(); k++){

				if(prediction_list[k].id == da_id){

					pred_x = prediction_list[k].x;
					pred_y = prediction_list[k].y;
				}
			}

			// new weight, using multivariate Gaussian
      double s_x = std_landmark[0];
      double s_y = std_landmark[1];
      double obs_w = (1/(2*M_PI*s_x*s_y)) * exp(-(pow(pred_x-obs_x, 2)/(2*pow(s_x, 2))
																								+(pow(pred_y-obs_y, 2)/(2*pow(s_y, 2)))));

			// save the weight update as the product of the new weight and the observations' weights
      particles[i].weight *= obs_w;
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// var to get all of the weights and to store the new particles
  vector<double> curr_w;
	vector<Particle> new_particles;

  for(int i=0; i<num_particles; i++){

    curr_w.push_back(particles[i].weight);
  }

  // generate a random int for the resampling init
  uniform_int_distribution<int> uni_d(0, num_particles-1);
  auto index_ = uni_d(gen);

  // get the largest current weight value and the max confusion
  double max_w = *max_element(curr_w.begin(), curr_w.end());
  uniform_real_distribution<double> uni_d_r(0., max_w);

  for(int i=0; i<num_particles; i++){

    double init_val = 0.+uni_d_r(gen)*2.;

		// keep iterating over the particles till the init_val is more than the current weight for said particle
    while (init_val > curr_w[index_]){

      init_val -= curr_w[index_];
      index_ = ((index_+1)%num_particles);
    }

		// add the new partivle to the list
    new_particles.push_back(particles[index_]);
  }

	// replace the old with the new
  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
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
