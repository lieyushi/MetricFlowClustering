#ifndef _METRIC_H
#define _METRIC_H

#include <eigen3/Eigen/Dense>
#include <climits>
#include <float.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include <algorithm>
#include <sys/time.h>
using namespace std;
using namespace Eigen;


struct MultiVariate
{	
	Matrix3f covariance;
	Vector3f meanVec;
	MultiVariate()
	{}
	~MultiVariate()
	{}
};


const float getNorm(const Eigen::VectorXf& r1, 
					const Eigen::VectorXf& r2, 
					const int& normOption);

void computeMeanRotation(float **data, 
						 const int& Row, 
						 const int& Column, 
						 std::vector<float>& rotation);

const float getRotation(float *array, 
						const int& size);

const float getRotation(const VectorXf& array, 
					    const int& size);

void getRotationSequence(float **data, 
						 const int& Row, 
						 const int& Column, 
						 std::vector<std::vector<float> >&rotationSequence);

void getSequence(float* array,
				 const int& size, 
				 std::vector<float>& rowSequence);

void getSequence(const VectorXf& array, 
				 const int& size, 
				 std::vector<float>& rowSequence);

void getNormalSequence(float **data, 
					   const int& Row, 
					   const int& Column, 
					   std::vector<MultiVariate>& normalMultivariate);

void getNormalMultivariate(float* array, 
				 	 	   const int& size, 
				 	 	   MultiVariate& rowSequence);

void getNormalMultivariate(const VectorXf& array, 
				 	 	   const int& size, 
				 	 	   MultiVariate& rowSequence);

void getFixedSequence(float **data, 
					  const int& Row, 
					  const int& Column, 
					  std::vector<std::vector<float> >&rotationSequence);

void getEachFixedSequence(float* array, 
				 		  const int& size, 
				 		  std::vector<float>& rowSequence);

void getEachFixedSequence(const VectorXf& array, 
				 		  const int& size, 
				 		  std::vector<float>& rowSequence);

void getUnnormalizedSequence(float **data, 
					   		 const int& Row, 
					  		 const int& Column, 
					  		 std::vector<MultiVariate>& normalMultivariate);

void getUnnormalizedMultivariate(float* array, 
				 	 	   		 const int& size, 
				 	 	  		 MultiVariate& rowSequence);

void getUnnormalizedMultivariate(const VectorXf& array, 
				 	 	  		 const int& size, 
				 	 	  		 MultiVariate& rowSequence);

#endif
