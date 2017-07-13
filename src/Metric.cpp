#include "Metric.h"

const float getNorm(const Eigen::VectorXf& r1, 
					const Eigen::VectorXf& r2, 
					const int& normOption)
{
	assert(r1.size()==r2.size());
	float length = 0.0;
	switch(normOption)
	{
	case 0:
	default:
		length = (r1-r2).norm();
		break;

	case 1:  /* fraction norm by high-dimensional feature-space */
		{
			for (int i = 0; i < r1.size(); ++i)
			{
				length += pow(abs(r1(i)-r2(i)),0.5);
			}
			length = pow(length,2.0);
		}
		break;

	/*case 2:  // the reason why it's commented is because sin could only result in angle
				  in range [0,pi/2]
		{
			const int& pointNum = r1.size()/3-1;
			float crossValue, leftNorm, rightNorm;
			Vector3f left, right;
			for (int i = 0; i < pointNum; ++i)
			{
				left << r1(3*i+3)-r1(3*i),r1(3*i+4)-r1(3*i+1),r1(3*i+5)-r1(3*i+2);
				right << r2(3*i+3)-r2(3*i),r2(3*i+4)-r2(3*i+1),r2(3*i+5)-r2(3*i+2);
				crossValue = (left.cross(right)).norm();
				leftNorm = left.norm(), rightNorm = right.norm();
				if(leftNorm >= 1.0e-6 && rightNorm >=1.0e-6)
					length+=asin(crossValue/leftNorm/rightNorm);
				else
					length+=M_PI/2.0;
			}
			length /= pointNum;
		}
		break;*/

	case 2: /* mean value of dot product value, which means it's rotational invariant */
		{
			const int& pointNum = r1.size()/3-1;
			float dotValue, leftNorm, rightNorm, result;
			Vector3f left, right;
			for (int i = 0; i < pointNum; ++i)
			{
				left << r1(3*i+3)-r1(3*i),r1(3*i+4)-r1(3*i+1),r1(3*i+5)-r1(3*i+2);
				right << r2(3*i+3)-r2(3*i),r2(3*i+4)-r2(3*i+1),r2(3*i+5)-r2(3*i+2);
				dotValue = left.dot(right);
				leftNorm = left.norm(), rightNorm = right.norm();
				if(leftNorm >= 1.0e-6 && rightNorm >=1.0e-6)
				{
					result = dotValue/leftNorm/rightNorm;
					result = min(1.0,(double)result);
					result = max(-1.0,(double)result);
					length+=acos(result);
				}
				else
					length+=M_PI/2.0;
			}
			length /= pointNum;
		}
		break;

	case 5: /* rotational invariant line-wise acos angle with normal direction for
			   measuring whether counterclockwise or clockwise orientation */
		{
			const int& pointNum = r1.size()/3-1;
			float dotValue, leftNorm, rightNorm, normalDot, result;
			Vector3f left, right, normal;

			left << r1(3)-r1(0),r1(4)-r1(1),r1(5)-r1(2);
			right << r2(3)-r2(0),r2(4)-r2(1),r2(5)-r2(2);
			const Vector3f& Normal = left.cross(right);

			for (int i = 0; i < pointNum; ++i)
			{
				left << r1(3*i+3)-r1(3*i),r1(3*i+4)-r1(3*i+1),r1(3*i+5)-r1(3*i+2);
				right << r2(3*i+3)-r2(3*i),r2(3*i+4)-r2(3*i+1),r2(3*i+5)-r2(3*i+2);
				normal = left.cross(right);
				dotValue = left.dot(right);
				normalDot = Normal.dot(normal);

				leftNorm = left.norm(), rightNorm = right.norm();
				if(leftNorm >= 1.0e-6 && rightNorm >=1.0e-6)
				{
					result = dotValue/leftNorm/rightNorm;
					result = min(1.0,(double)result);
					result = max(-1.0,(double)result);
					if(normalDot<0)
						length+=-acos(result);
					else
						length+=acos(result);
				}
				else
					length+=M_PI/2.0;
			}
			length /= pointNum;
		}
		break;	

	case 8: /* distance metric defined as mean * standard deviation */
		{
			const int& pointNum = r1.size()/3-1;
			float dotValue, leftNorm, rightNorm, stdevia = 0.0, angle, result;
			Vector3f left, right;
			for (int i = 0; i < pointNum; ++i)
			{
				left << r1(3*i+3)-r1(3*i),r1(3*i+4)-r1(3*i+1),r1(3*i+5)-r1(3*i+2);
				right << r2(3*i+3)-r2(3*i),r2(3*i+4)-r2(3*i+1),r2(3*i+5)-r2(3*i+2);
				dotValue = left.dot(right);
				leftNorm = left.norm(), rightNorm = right.norm();
				if(leftNorm >= 1.0e-6 && rightNorm >=1.0e-6)
				{
					result = dotValue/leftNorm/rightNorm;
					result = min(1.0,(double)result);
					result = max(-1.0,(double)result);
					angle = acos(result);
					length+=angle;
					stdevia+=angle*angle;
				}
				else
				{
					angle=M_PI/2.0;
					length+=angle;
					stdevia+=angle*angle;
				}
			}
			length /= pointNum;
			stdevia = sqrt(stdevia/pointNum-length*length);
			length*=stdevia;
		}
		break;

	case 10: /* generalized cross dot product */
		{
			const int& size = r1.size()/3;
			VectorXf x(r1.size()), y(r2.size());
			Vector3f left, right;
			float leftNorm, rightNorm;
			for (int i = 0; i < size; ++i)
			{
				left << r1(3*i), r1(3*i+1), r1(3*i+2);
				right << r2(3*i), r2(3*i+1), r2(3*i+2);
				leftNorm = left.norm();
				rightNorm = right.norm();
				// I Know it's hardly possible to have smaller norm, but just in case
				if(leftNorm>1.0e-6)
				{
					for (int j = 0; j < 3; ++j)
					{
						x(3*i+j) = left(j)/leftNorm;
					}
				}
				//norm to zero, choose the largest to be 1
				else
				{
					for (int j = 0; j < 3; ++j)
					{
						x(3*i+j) = 0;
					}
				}

				if(rightNorm>1.0e-6)
				{
					for (int j = 0; j < 3; ++j)
					{
						y(3*i+j) = right(j)/rightNorm;
					}
				}
				else
				{
					for (int j = 0; j < 3; ++j)
					{
						y(3*i+j) = 0;
					}
				}
			}
			length = x.dot(y)/r1.size();
			length = min(1.0,(double)length);
			length = max(-1.0,(double)length);
			length = acos(length);

			/*float result = r1.dot(r2);
			float firstLength = r1.norm();
			float secondLength = r2.norm();
			if(firstLength>1.0e-6)
				result/=firstLength;
			if(secondLength>1.0e-6)
				result/=secondLength;
			result = min(1.0,(double)result);
			result = max(-1.0,(double)result);
			length = acos(length);*/
		}
		break;
	}

	return abs(length);
}


void computeMeanRotation(float **data, 
						 const int& Row, 
						 const int& Column, 
						 std::vector<float>& rotation)
{
	rotation = std::vector<float>(Row, 0.0);
	const int& pointNum = Column/3-2;
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int i = 0; i < Row; ++i)
	{
		rotation[i] = getRotation(data[i], pointNum);
	}
}


const float getRotation(float *array, 
						const int& size)
{
	float dotValue, leftNorm, rightNorm, meanRotation = 0.0, result;
	Vector3f left, right;
	for (int j = 0; j < size; ++j)
	{
		left << array[j*3+3]-array[j*3], array[j*3+4]-array[j*3+1], array[j*3+5]-array[j*3+2];
		right << array[j*3+6]-array[j*3+3], array[j*3+7]-array[j*3+4], array[j*3+8]-array[j*3+5];
		dotValue = left.dot(right);
		leftNorm = left.norm();
		rightNorm = right.norm();
		if(leftNorm >= 1.0e-6 && rightNorm >=1.0e-6)
		{
			result = dotValue/leftNorm/rightNorm;
			result = min(1.0,(double)result);
			result = max(-1.0,(double)result);
			meanRotation += acos(result);
		}
	}
	meanRotation/=size;
	return meanRotation;
}


const float getRotation(const VectorXf& array, 
						const int& size)
{
	float dotValue, leftNorm, rightNorm, meanRotation = 0.0, result;
	Vector3f left, right;
	for (int j = 0; j < size; ++j)
	{
		left << array(j*3+3)-array(j*3), array(j*3+4)-array(j*3+1), array(j*3+5)-array(j*3+2);
		right << array(j*3+6)-array(j*3+3), array(j*3+7)-array(j*3+4), array(j*3+8)-array(j*3+5);
		dotValue = left.dot(right);
		leftNorm = left.norm();
		rightNorm = right.norm();
		if(leftNorm >= 1.0e-6 && rightNorm >=1.0e-6)
		{
			result = dotValue/leftNorm/rightNorm;
			result = min(1.0,(double)result);
			result = max(-1.0,(double)result);
			meanRotation += acos(result);
		}
	}
	meanRotation/=size;
	return meanRotation;
}


void getRotationSequence(float **data, 
						 const int& Row, 
						 const int& Column, 
						 std::vector<std::vector<float> >&rotationSequence)
{
	const int& pointNum = Column/3-2;
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int i = 0; i < Row; ++i)
	{
		getSequence(data[i], pointNum, rotationSequence[i]);
	}

	ofstream fout("mean_std.txt",ios::out);
	if(!fout)
	{
		std::cout << "Error opening this file!" << std::endl;
		exit(1);
	}
	for (int i = 0; i < Row; ++i)
	{
		fout << rotationSequence[i][0] << " " << rotationSequence[i][1] << std::endl;
	}
	fout.close();
}


void getSequence(float* array, 
				 const int& size, 
				 std::vector<float>& rowSequence)
{
	rowSequence = std::vector<float>(2);
	float dotValue, leftNorm, rightNorm, meanRotation = 0.0, deviation = 0.0, angle, result;
	Vector3f left, right;
	for (int j = 0; j < size; ++j)
	{
		left << array[j*3+3]-array[j*3], array[j*3+4]-array[j*3+1], array[j*3+5]-array[j*3+2];
		right << array[j*3+6]-array[j*3+3], array[j*3+7]-array[j*3+4], array[j*3+8]-array[j*3+5];
		dotValue = left.dot(right);
		leftNorm = left.norm();
		rightNorm = right.norm();
		if(leftNorm >= 1.0e-6 && rightNorm >=1.0e-6)
		{
			result = dotValue/leftNorm/rightNorm;
			/* clamp acos(x) to be [-1.0,1.0] */
			result = min(1.0,(double)result);
			result = max(-1.0,(double)result);
			angle = acos(result);
			meanRotation += angle;
			deviation += angle*angle;
		}
	}
	meanRotation /= size;
	rowSequence[0] = meanRotation;
	result = deviation/size-meanRotation*meanRotation;
	rowSequence[1] = sqrt(result);
}


void getSequence(const VectorXf& array, 
				 const int& size, 
				 std::vector<float>& rowSequence)
{
	rowSequence = std::vector<float>(2);
	float dotValue, leftNorm, rightNorm, meanRotation = 0.0, deviation = 0.0, angle, result;
	Vector3f left, right;
	for (int j = 0; j < size; ++j)
	{
		left << array(j*3+3)-array(j*3), array(j*3+4)-array(j*3+1), array(j*3+5)-array(j*3+2);
		right << array(j*3+6)-array(j*3+3), array(j*3+7)-array(j*3+4), array(j*3+8)-array(j*3+5);
		dotValue = left.dot(right);
		leftNorm = left.norm();
		rightNorm = right.norm();
		if(leftNorm >= 1.0e-6 && rightNorm >=1.0e-6)
		{
			result = dotValue/leftNorm/rightNorm;
			result = min(1.0,(double)result);
			result = max(-1.0,(double)result);
			angle = acos(result);
			meanRotation += angle;
			deviation += angle*angle;
		}
	}
	meanRotation /= size;
	rowSequence[0] = meanRotation;
	rowSequence[1] = sqrt(deviation/size-(meanRotation*meanRotation));
}


void getNormalSequence(float **data, 
					   const int& Row, 
					   const int& Column, 
					   std::vector<MultiVariate>& normalMultivariate)
{
	const int& pointNum = Column/3-1;
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int i = 0; i < Row; ++i)
	{
		getNormalMultivariate(data[i], pointNum, normalMultivariate[i]);
	}
}


void getNormalMultivariate(float* array, 
				 	 	   const int& size, 
				 	 	   MultiVariate& rowSequence)
{
	MatrixXf normalDirection(size,3);
	float leftNorm;
	Vector3f left;
	VectorXf unitOne(size);
	for (int j = 0; j < size; ++j)
	{
		left << array[j*3+3]-array[j*3], array[j*3+4]-array[j*3+1], array[j*3+5]-array[j*3+2];
		leftNorm = left.norm();
		if(leftNorm >= 1.0e-8)
		{
			for(int k=0;k<3;k++)
				/* record each line segment normal direction */
				normalDirection(j,k) = left(k)/leftNorm;
		}
		else
		{
			for(int k=0;k<3;k++)
				/* if norm is small, mark them as zero to tell identical points */
				normalDirection(j,k) = 0.0;
		}
		unitOne(j) = 1.0;
	}

	VectorXf meanNormal(3);
	for (int i = 0; i < 3; ++i)
	{
		meanNormal(i) = normalDirection.transpose().row(i).mean();
	}

	MatrixXf tempMatrix = normalDirection-unitOne*meanNormal.transpose();
	rowSequence.covariance = tempMatrix.transpose()*tempMatrix/(size-1);
	rowSequence.meanVec = meanNormal;
}


void getNormalMultivariate(const VectorXf& array, 
				 	 	   const int& size, 
				 	 	   MultiVariate& rowSequence)
{
	MatrixXf normalDirection(size,3);
	float leftNorm;
	Vector3f left;
	VectorXf unitOne(size);
	for (int j = 0; j < size; ++j)
	{
		left << array(j*3+3)-array(j*3), array(j*3+4)-array(j*3+1), array(j*3+5)-array(j*3+2);
		leftNorm = left.norm();
		if(leftNorm >= 1.0e-8)
		{
			for(int k=0;k<3;k++)
				/* record each line segment normal direction */
				normalDirection(j,k) = left(k)/leftNorm;
		}
		else
		{
			for(int k=0;k<3;k++)
				/* if norm is small, mark them as zero to tell identical points */
				normalDirection(j,k) = 0.0;
		}
		unitOne(j) = 1.0;
	}

	VectorXf meanNormal(3);
	for (int i = 0; i < 3; ++i)
	{
		meanNormal(i) = normalDirection.transpose().row(i).mean();
	}

	MatrixXf tempMatrix = normalDirection-unitOne*meanNormal.transpose();
	rowSequence.covariance = tempMatrix.transpose()*tempMatrix/(size-1);
	rowSequence.meanVec = meanNormal;
}


void getFixedSequence(float **data, 
					  const int& Row, 
					  const int& Column, 
					  std::vector<std::vector<float> >&rotationSequence)
{
	const int& pointNum = Column/3-1;
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int i = 0; i < Row; ++i)
	{
		getEachFixedSequence(data[i], pointNum, rotationSequence[i]);
	}
}


void getEachFixedSequence(float* array, 
				 		  const int& size, 
				 		  std::vector<float>& rowSequence)
{
	rowSequence = std::vector<float>(2);
	float dotValue, leftNorm, meanRotation = 0.0, deviation = 0.0, angle, result;
	Vector3f left, xRay;
	xRay << 1.0,0.0,0.0;
	for (int j = 0; j < size; ++j)
	{
		left << array[j*3+3]-array[j*3], array[j*3+4]-array[j*3+1], array[j*3+5]-array[j*3+2];
		dotValue = left.dot(xRay);
		leftNorm = left.norm();
		if(leftNorm >= 1.0e-6)
		{
			result = dotValue/leftNorm;
			result = min(1.0,(double)result);
			result = max(-1.0,(double)result);
			angle = acos(result);
			meanRotation += angle;
			deviation += angle*angle;
		}
	}
	meanRotation /= size;
	rowSequence[0] = meanRotation;
	rowSequence[1] = sqrt(deviation/size-(meanRotation*meanRotation));
}


void getEachFixedSequence(const VectorXf& array, 
				 		  const int& size, 
				 		  std::vector<float>& rowSequence)
{
	rowSequence = std::vector<float>(2);
	float dotValue, leftNorm, meanRotation = 0.0, deviation = 0.0, angle, result;
	Vector3f left, xRay;
	xRay << 1.0,0.0,0.0;
	for (int j = 0; j < size; ++j)
	{
		left << array(j*3+3)-array(j*3), array(j*3+4)-array(j*3+1), array(j*3+5)-array(j*3+2);
		dotValue = left.dot(xRay);
		leftNorm = left.norm();
		if(leftNorm >= 1.0e-6)
		{
			result = dotValue/leftNorm;
			result = min(1.0,(double)result);
			result = max(-1.0,(double)result);
			angle = acos(result);
			meanRotation += angle;
			deviation += angle*angle;
		}
	}
	meanRotation /= size;
	rowSequence[0] = meanRotation;
	rowSequence[1] = sqrt(deviation/size-(meanRotation*meanRotation));
}


void getUnnormalizedSequence(float **data, 
					   		 const int& Row, 
					  		 const int& Column, 
					  		 std::vector<MultiVariate>& normalMultivariate)
{
	const int& pointNum = Column/3-1;
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int i = 0; i < Row; ++i)
	{
		getUnnormalizedMultivariate(data[i], pointNum, normalMultivariate[i]);
	}
}



void getUnnormalizedMultivariate(float* array, 
				 	 	   		 const int& size, 
				 	 	  		 MultiVariate& rowSequence)
{
	MatrixXf normalDirection(size,3);
	float leftNorm;
	Vector3f left;
	VectorXf unitOne(size);
	for (int j = 0; j < size; ++j)
	{
		left << array[j*3+3]-array[j*3], array[j*3+4]-array[j*3+1], array[j*3+5]-array[j*3+2];
		leftNorm = left.norm();
		if(leftNorm >= 1.0e-8)
		{
			for(int k=0;k<3;k++)
				/* record each line segment normal direction */
				normalDirection(j,k) = left(k);
		}
		else
		{
			for(int k=0;k<3;k++)
				/* if norm is small, mark them as zero to tell identical points */
				normalDirection(j,k) = 0.0;
		}
		unitOne(j) = 1.0;
	}

	VectorXf meanNormal(3);
	for (int i = 0; i < 3; ++i)
	{
		meanNormal(i) = normalDirection.transpose().row(i).mean();
	}

	MatrixXf tempMatrix = normalDirection-unitOne*meanNormal.transpose();
	rowSequence.covariance = tempMatrix.transpose()*tempMatrix/(size-1);
	rowSequence.meanVec = meanNormal;
}


void getUnnormalizedMultivariate(const VectorXf& array, 
				 	 	  		 const int& size, 
				 	 	  		 MultiVariate& rowSequence)
{
	MatrixXf normalDirection(size,3);
	float leftNorm;
	Vector3f left;
	VectorXf unitOne(size);
	for (int j = 0; j < size; ++j)
	{
		left << array(j*3+3)-array(j*3), array(j*3+4)-array(j*3+1), array(j*3+5)-array(j*3+2);
		leftNorm = left.norm();
		if(leftNorm >= 1.0e-8)
		{
			for(int k=0;k<3;k++)
				/* record each line segment normal direction */
				normalDirection(j,k) = left(k);
		}
		else
		{
			for(int k=0;k<3;k++)
				/* if norm is small, mark them as zero to tell identical points */
				normalDirection(j,k) = 0.0;
		}
		unitOne(j) = 1.0;
	}

	VectorXf meanNormal(3);
	for (int i = 0; i < 3; ++i)
	{
		meanNormal(i) = normalDirection.transpose().row(i).mean();
	}

	MatrixXf tempMatrix = normalDirection-unitOne*meanNormal.transpose();
	rowSequence.covariance = tempMatrix.transpose()*tempMatrix/(size-1);
	rowSequence.meanVec = meanNormal;
}
