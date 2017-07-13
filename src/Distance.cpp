#include "Distance.h"


/* ------------------ Compute norm 3 for trajectories ------------------------- */
// given a center trajectory and index of pre-stored vector 
const float getBMetric_3(const VectorXf& row,
						 const int& size,
						 const int& i,
						 const std::vector<std::vector<float> >& rotationSequence
						)
{
	std::vector<float> firstNorm3, secondNorm3;
	getSequence(row, size, firstNorm3);
	secondNorm3 = rotationSequence[i];
	return getBMetric(firstNorm3, secondNorm3);
}

// given two center trajectories for distance measuring 
const float getBMetric_3(const VectorXf& firstRow,
						 const int& size,
						 const VectorXf& secondRow
						)
{
	std::vector<float> firstNorm3, secondNorm3;
	getSequence(firstRow, size, firstNorm3);
	getSequence(secondRow, size, secondNorm3);
	return getBMetric(firstNorm3, secondNorm3);
}
/* -------------------- Finish computing norm 3 for trajectories --------------------*/

/* ------------------ Compute norm 6 for trajectories ------------------------- */
// given a center trajectory and index of pre-stored vector 
const float getBMetric_6(const VectorXf& row,
						 const int& size,
						 const int& i,
						 const std::vector<MultiVariate>& normalMultivariate
						)
{
	MultiVariate centerNormal, neighNormal;
	getNormalMultivariate(row, size, centerNormal);
	neighNormal = normalMultivariate[i];
	return getBMetric(centerNormal, neighNormal);
}

// given two center trajectories for distance measuring 
const float getBMetric_6(const VectorXf& firstRow,
						 const int& size,
						 const VectorXf& secondRow
						)
{
	MultiVariate centerNormal, neighNormal;
	getNormalMultivariate(firstRow, size, centerNormal);
	getNormalMultivariate(secondRow, size, neighNormal);
	return getBMetric(centerNormal, neighNormal);
}
/* -------------------- Finish computing norm 6 for trajectories --------------------*/


/* ------------------ Compute norm 7 for trajectories ------------------------- */
// given a center trajectory and index of pre-stored vector 
const float getBMetric_7(const VectorXf& row,
						 const int& size,
						 const int& i,
						 const std::vector<std::vector<float> >& rotationSequence
						)
{
	std::vector<float> firstNorm3, secondNorm3;
	getEachFixedSequence(row, size, firstNorm3);
	secondNorm3 = rotationSequence[i];
	return getBMetric(firstNorm3, secondNorm3);
}

// given two center trajectories for distance measuring 
const float getBMetric_7(const VectorXf& firstRow,
						 const int& size,
						 const VectorXf& secondRow
						)
{
	std::vector<float> firstNorm3, secondNorm3;
	getEachFixedSequence(firstRow, size, firstNorm3);
	getEachFixedSequence(secondRow, size, secondNorm3);
	return getBMetric(firstNorm3, secondNorm3);
}
/* -------------------- Finish computing norm 7 for trajectories --------------------*/


/* ------------------ Compute norm 9 for trajectories ------------------------- */
// given a center trajectory and index of pre-stored vector 
const float getBMetric_9(const VectorXf& row,
						 const int& size,
						 const int& i,
						 const std::vector<MultiVariate>& normalMultivariate
						)
{
	MultiVariate centerNormal, neighNormal;
	getUnnormalizedMultivariate(row, size, centerNormal);
	neighNormal = normalMultivariate[i];
	return getBMetric(centerNormal, neighNormal);
}

// given two center trajectories for distance measuring 
const float getBMetric_9(const VectorXf& firstRow,
						 const int& size,
						 const VectorXf& secondRow
						)
{
	MultiVariate centerNormal, neighNormal;
	getUnnormalizedMultivariate(firstRow, size, centerNormal);
	getUnnormalizedMultivariate(secondRow, size, neighNormal);
	return getBMetric(centerNormal, neighNormal);
}
/* -------------------- Finish computing norm 9 for trajectories --------------------*/



const float getBMetric(const std::vector<float>& firstNorm3, 
					   const std::vector<float>& secondNorm3
					  )
{
	float u_a, u_b, sig_a, sig_b, sig_a_inverse, sig_b_inverse, 
		  summation, sum_inverse, tempDist;
	u_a = firstNorm3[0], u_b = secondNorm3[0];
	sig_a = firstNorm3[1], sig_b = secondNorm3[1];
	sig_a_inverse = sig_a>1.0e-6?1.0/sig_a:0;
	sig_b_inverse = sig_b>1.0e-6?1.0/sig_b:0;
	summation = sig_a*sig_a+sig_b*sig_b;
	sum_inverse = summation>1.0e-6?1.0/summation:0.0;
	tempDist = 0.25*log(0.25*(sig_a*sig_a*sig_b_inverse*sig_b_inverse
			   +sig_b*sig_b*sig_a_inverse*sig_a_inverse+2))
			   + 0.25*(u_a-u_b)*(u_a-u_b)*sum_inverse;
	return tempDist;
}


const float getBMetric(const MultiVariate& centerNormal, 
					   const MultiVariate& neighNormal
					  ) 
{
	Matrix3f firstCov, secondCov, meanCov, meanCovInverse;
	float sqrtInverse;
	firstCov = centerNormal.covariance;
	secondCov = neighNormal.covariance;
	meanCov = 0.5*(firstCov+secondCov);
	if(meanCov.determinant()>1.0e-6)
		meanCovInverse = static_cast<Matrix3f>(meanCov.inverse());
	else
		meanCovInverse = meanCov;
	float detMulti = sqrt(firstCov.determinant()*secondCov.determinant());
	sqrtInverse = detMulti>1.0e-6?float(1.0)/detMulti:float(0);
	Vector3f meanDiff = centerNormal.meanVec-neighNormal.meanVec;
	float tempDist = 0.125*meanDiff.transpose()*meanCovInverse*meanDiff
			   +0.2*log(meanCov.determinant()*sqrtInverse);
	return tempDist;
}