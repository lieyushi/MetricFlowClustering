#include "PCA_Cluster.h"

const float& TOR_1 = 0.999;
const int& CLUSTER = 8;

extern int initializationOption;

void PCA_Cluster::performPCA_Clustering(float **data, 
										const int& Row, 
										const int& Column, 
										std::vector<MeanLine>& massCenter,
									    std::vector<int>& group, 
									    std::vector<int>& totalNum, 
									    std::vector<ExtractedLine>& closest,
									    std::vector<ExtractedLine>& furthest,
									    float& entropy)
{
	MatrixXf cArray, SingVec;
	VectorXf meanTrajectory(Column);
	int PC_Number;

	performSVD(cArray, data, Row, Column, PC_Number, SingVec, meanTrajectory);
	performPC_KMeans(cArray, Row, Column, PC_Number, SingVec, meanTrajectory, 
					 massCenter, CLUSTER, group, totalNum, closest, 
					 furthest, data, entropy);
}



void PCA_Cluster::performSVD(MatrixXf& cArray, 
							 float **data, 
							 const int& Row, 
							 const int& Column,
							 int& PC_Number, 
							 MatrixXf& SingVec, 
							 VectorXf& meanTrajectory)
{
	Eigen::MatrixXf temp(Row, Column);
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int i = 0; i < Row; ++i)
	{
		temp.row(i) = Eigen::VectorXf::Map(&data[i][0], Column); //copy each trajectory to temporary space
	}
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int i = 0; i < Column; ++i)
	{
		meanTrajectory(i) = temp.transpose().row(i).mean();
	}
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int i = 0; i < Row; ++i)
	{
		temp.row(i) = temp.row(i)-meanTrajectory.transpose();
	}

	struct timeval start, end;
	gettimeofday(&start, NULL);
	/* perform SVD decomposition for temp */
	JacobiSVD<MatrixXf> svd(temp, ComputeThinU | ComputeThinV);
	//const VectorXf& singValue = svd.singularValues();
	SingVec = svd.matrixV();
	gettimeofday(&end, NULL);
	const double& delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;

	std::cout << "SVD decomposition takes " <<delta << " s!" << std::endl;
	/* compute new attribute space based on principal component */
	MatrixXf coefficient = temp*SingVec;
	/*  decide first r dorminant PCs with a threshold */
	const float& varianceSummation = coefficient.squaredNorm();
	float tempSum = 0.0;
	const float& threshold = TOR_1*varianceSummation;
	
	for (int i = 0; i < Column; ++i)
	{
		tempSum+=(coefficient.transpose().row(i)).squaredNorm();
		if(tempSum>threshold)
		{
			PC_Number = i;
			break;
		}
	}

	cArray = MatrixXf(Row, PC_Number);
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int i = 0; i < PC_Number; ++i)
	{
		cArray.transpose().row(i) = coefficient.transpose().row(i);
	}

	std::cout << "SVD completed!" << std::endl;

	SingVec.transposeInPlace();
}



void PCA_Cluster::performPC_KMeans(const MatrixXf& cArray, 
								   const int& Row, 
								   const int& Column, 
								   const int& PC_Number, 
				 				   const MatrixXf& SingVec, 
				 				   const VectorXf& meanTrajectory, 
				 				   std::vector<MeanLine>& massCenter, 
				 				   const int& Cluster, 
				 				   std::vector<int>& group, 
				 				   std::vector<int>& totalNum, 
				 				   std::vector<ExtractedLine>& closest,
				 				   std::vector<ExtractedLine>& furthest, 
				 				   float **data,
				 				   float& entropy)
{
/* perform K-means clustering */
	MatrixXf clusterCenter;

	switch(initializationOption)
	{
	case 1:
		Initialization::generateRandomPos(clusterCenter, PC_Number, cArray, Cluster);
		break;

	case 2:
		Initialization::generateFromSamples(clusterCenter, PC_Number, cArray, Cluster);
		break;

	case 3:
		Initialization::generateFarSamples(clusterCenter, PC_Number, cArray, Cluster, 0);
		break;
	}

	std::cout << "Initialization completed!" << std::endl;

	float moving=100, tempMoving, before;
	int storage[Cluster];

	MatrixXf centerTemp;  //store provisional center coordinate

	int tag = 0, clusTemp;

	float temp;

	std::vector< std::vector<int> > neighborVec(Cluster, std::vector<int>());

	double PCA_KMeans_delta, KMeans_delta;
	struct timeval start, end;

	gettimeofday(&start, NULL);

	std::vector<int> recorder(Row);
	do
	{
		before = moving;
		/* preset cluster number recorder */
		memset(storage,0,sizeof(int)*Cluster);
		centerTemp = MatrixXf::Zero(Cluster, PC_Number);

	#pragma omp parallel for schedule(dynamic) num_threads(8)
		for (int i = 0; i < Cluster; ++i)
		{
			neighborVec[i].clear();
		}

	//#pragma omp parallel for reduction(+:storage,centerTemp) nowait num_threads(8)
		for (int i = 0; i < Row; ++i)
		{
			float dist = FLT_MAX;
			for (int j = 0; j < Cluster; ++j)
			{
				temp = (cArray.row(i)-clusterCenter.row(j)).norm();
				if(temp<dist)
				{
					dist = temp;
					clusTemp = j;
				}
			}
			storage[clusTemp]++;
			neighborVec[clusTemp].push_back(i);
			recorder[i] = clusTemp;
			centerTemp.row(clusTemp)+=cArray.row(i);
		}

		moving = FLT_MIN;

	#pragma omp parallel for reduction(max:moving) num_threads(8)
		for (int i = 0; i < Cluster; ++i)
		{
			if(storage[i]>0)
			{
				centerTemp.row(i)/=storage[i];
				tempMoving = (centerTemp.row(i)-clusterCenter.row(i)).norm();
				clusterCenter.row(i) = centerTemp.row(i);
				if(moving<tempMoving)
					moving = tempMoving;
			}
		}
		std::cout << "K-means iteration " << ++tag << " completed, and moving is " 
		<< moving << "!" << std::endl;
	}while(abs(moving-before)/before >= 1.0e-2 && tag <= 20);

	gettimeofday(&end, NULL);
	
	const double& delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;

	std::cout << "K-Means for PC takes " << delta << " s!" << std::endl;

	std::multimap<int,int> groupMap;

	entropy = 0.0;
	float probability;
	for (int i = 0; i < Cluster; ++i)
	{
		groupMap.insert(std::pair<int,int>(storage[i],i));
		if(storage[i]>0)
		{
			probability = float(storage[i])/float(Row);
			entropy += probability*log(probability);
		}
	}
	entropy = -entropy;

	int groupNo = 0;
	int increasingOrder[Cluster];
	for (multimap<int,int>::iterator it = groupMap.begin(); it != groupMap.end(); ++it)
	{
		if(it->first>0)
		{
			increasingOrder[it->second] = (groupNo++);
		}
	}

#pragma omp parallel for schedule(dynamic) num_threads(8)	
	for (int i = 0; i < Row; ++i)
	{
		group[i] = increasingOrder[recorder[i]];
		totalNum[i] = storage[recorder[i]];
	}

	float shortest, farDist, toCenter;
	int shortestIndex = 0, fartestIndex = 0, tempIndex = 0;
	std::vector<int> neighborTemp;

	for (int i = 0; i < Cluster; ++i)
	{
		if(storage[i]>0 && !neighborVec[i].empty())
		{
			neighborTemp = neighborVec[i];
			shortest = FLT_MAX;
			farDist = FLT_MIN;

			for (int j = 0; j < storage[i]; ++j)
			{
				tempIndex = neighborTemp[j];
				toCenter = (clusterCenter.row(i)-cArray.row(tempIndex)).norm();

				if(toCenter<shortest)
				{
					shortest = toCenter;
					shortestIndex = tempIndex;
				}
				if(toCenter>farDist)
				{
					farDist = toCenter;
					fartestIndex = tempIndex;
				}
			}
			closest.push_back(ExtractedLine(shortestIndex,increasingOrder[i]));
			furthest.push_back(ExtractedLine(fartestIndex,increasingOrder[i]));
		}
	}

	MatrixXf pcSing(PC_Number,Column);

#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int i = 0; i < PC_Number; ++i)
	{
		pcSing.row(i) = SingVec.row(i);
	}

	MatrixXf massPos = clusterCenter*pcSing;

	for (int i = 0; i < Cluster; ++i)
	{
		if(storage[i]>0)
		{
			massPos.row(i) += meanTrajectory.transpose();
			std::vector<float> vecTemp;
			for (int j = 0; j < Column; ++j)
			{
				vecTemp.push_back(massPos(i,j));
			}
			massCenter.push_back(MeanLine(vecTemp,increasingOrder[i]));
		}
	}

	std::cout << "Mean position re-mapped to trajectory space completed!" << std::endl;
}


void PCA_Cluster::performAHC(const MatrixXf& cArray, 
							 const int& Row, 
							 const int& Column, 
							 const int& PC_Number, 
		      				 const MatrixXf& SingVec, 
		      				 const VectorXf& meanTrajectory, 
		      				 MatrixXf& clusterCenter, 
		      				 std::vector<MeanLine>& massCenter)
{
	return;
}


void PCA_Cluster::performDirectK_Means(float **data, 
									   const int& Row, 
									   const int& Column, 
									   std::vector<MeanLine>& massCenter,
									   std::vector<int>& group, 
									   std::vector<int>& totalNum, 
									   std::vector<ExtractedLine>& closest,
									   std::vector<ExtractedLine>& furthest, 
									   const int& normOption,
									   float& entropy)
{
	performFullK_MeansByClusters(data, Row, Column, massCenter, CLUSTER, group, 
								 totalNum, closest, furthest, normOption, entropy);
}


void PCA_Cluster::performPCA_Clustering(float **data, 
										const int& Row, 
										const int& Column, 
										std::vector<MeanLine>& massCenter,
										std::vector<int>& group, 
										std::vector<int>& totalNum, 
										std::vector<ExtractedLine>& closest, 
										std::vector<ExtractedLine>& furthest, 
										const int& Cluster,
										float& entropy)
{
	MatrixXf cArray, SingVec;
	VectorXf meanTrajectory(Column);
	int PC_Number;

	performSVD(cArray, data, Row, Column, PC_Number, SingVec, meanTrajectory);
	performPC_KMeans(cArray, Row, Column, PC_Number, SingVec, meanTrajectory, 
					 massCenter, Cluster, group, totalNum, closest,
					 furthest, data, entropy);
}


void PCA_Cluster::performDirectK_Means(float **data, 
									   const int& Row, 
									   const int& Column, 
									   std::vector<MeanLine>& massCenter,
									   std::vector<int>& group, 
									   std::vector<int>& totalNum, 
									   std::vector<ExtractedLine>& closest, 
									   std::vector<ExtractedLine>& furthest, 
									   const int& Cluster, 
									   const int& normOption,
									   float& entropy)
{
	performFullK_MeansByClusters(data, Row, Column, massCenter, Cluster, group, 
								 totalNum, closest, furthest, normOption, entropy);
}


void PCA_Cluster::performFullK_MeansByClusters(float **data, 
											   const int& Row, 
											   const int& Column, 
											   std::vector<MeanLine>& massCenter,
											   const int& Cluster, 
											   std::vector<int>& group, 
											   std::vector<int>& totalNum, 
											   std::vector<ExtractedLine>& closest, 
											   std::vector<ExtractedLine>& furthest, 
											   const int& normOption,
											   float& entropy)
{
	Eigen::MatrixXf temp(Row, Column);

#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int i = 0; i < Row; ++i)
	{
		temp.row(i) = Eigen::VectorXf::Map(&data[i][0], Column); //copy each trajectory to temporary space
	}

	MatrixXf clusterCenter;

	switch(initializationOption)
	{
	case 1:
		Initialization::generateRandomPos(clusterCenter, Column, temp, Cluster);
		break;

	case 2:
		Initialization::generateFromSamples(clusterCenter, Column, temp, Cluster);
		break;

	case 3:
		Initialization::generateFarSamples(clusterCenter, Column, temp, Cluster, normOption);
		break;
	}

	std::cout << "Initialization completed!" << std::endl;

	float moving=100, tempMoving, dist, tempDist, before;
	int *storage = new int[Cluster]; // used to store number inside each cluster
	MatrixXf centerTemp;
	int tag = 0, clusTemp;
	std::vector< std::vector<int> > neighborVec(Cluster, std::vector<int>());

	std::vector<float> rotation;
	std::vector<std::vector<float> > rotationSequence(Row,std::vector<float>(2));
	std::vector<MultiVariate> normalMultivariate(Row, MultiVariate());

/*  if rotation used for judge similarity difference, has to use pre-defined cache */	
	if(normOption==4)
	{
		computeMeanRotation(data, Row, Column, rotation);
	}
/*  end defining pre-ordered cache for rotation */

/*  pre-defined cache for sequence mean and standard deviation */
	else if(normOption==3)
	{
		getRotationSequence(data, Row, Column, rotationSequence);
	}
/*  finish computing sequence value */

	else if(normOption==6)
	{
		getNormalSequence(data, Row, Column, normalMultivariate);
	}

	else if(normOption==7)
	{
		getFixedSequence(data, Row, Column, rotationSequence);
	}

	else if(normOption==9)
	{
		getUnnormalizedSequence(data, Row, Column, normalMultivariate);
	}

/* perform K-means with different metrics */
	std::cout << "K-means start!" << std::endl;	
	struct timeval start, end;
	gettimeofday(&start, NULL);
	std::vector<int> recorder(Row); //use to record which cluster the row belongs to

	do
	{
	/* reset storage number and weighted mean inside each cluster*/
		before=moving;
		memset(storage,0,sizeof(int)*Cluster);
		centerTemp = MatrixXf::Zero(Cluster,Column);

	/* clear streamline indices for each cluster */
	#pragma omp parallel for schedule(dynamic) num_threads(8)
		for (int i = 0; i < Cluster; ++i)
		{
			neighborVec[i].clear();
		}

		for (int i = 0; i < Row; ++i)
		{
			dist = FLT_MAX;
			for (int j = 0; j < Cluster; ++j)
			{
				if(normOption==4)
					tempDist = abs(rotation[i]-getRotation(clusterCenter.row(j), Column/3-2));
				// this is the B-metric for two gaussian distribution
				else if(normOption==3)
					tempDist = getBMetric_3(clusterCenter.row(j),Column/3-2,i,rotationSequence);
				else if(normOption==6)
					tempDist = getBMetric_6(clusterCenter.row(j),Column/3-1,i,normalMultivariate);
				else if(normOption==7)
					tempDist = getBMetric_7(clusterCenter.row(j), Column/3-1,i,rotationSequence);
				else if(normOption==9)
					tempDist = getBMetric_9(clusterCenter.row(j),Column/3-1,i,normalMultivariate);
				else
					tempDist = getNorm(temp.row(i),clusterCenter.row(j),normOption);

				if(tempDist<dist)
				{
					dist = tempDist;
					clusTemp = j;
				}
			}
			recorder[i] = clusTemp;
			storage[clusTemp]++;
			neighborVec[clusTemp].push_back(i);
			centerTemp.row(clusTemp)+=temp.row(i);
		}
		moving = FLT_MIN;

	/* measure how much the current center moves from original center */	
	#pragma omp parallel for reduction(max:moving) num_threads(8)
		for (int i = 0; i < Cluster; ++i)
		{
			if(storage[i]>0)
			{
				centerTemp.row(i)/=storage[i];
				/*if(normOption==4)
					tempMoving = abs(getRotation(centerTemp.row(i), Column/3-2)-
									 getRotation(clusterCenter.row(i), Column/3-2));
				else if(normOption==3)
					tempMoving = getBMetric_3(centerTemp.row(i), Column/3-2, clusterCenter.row(i));
				else if(normOption==6)
					tempMoving = getBMetric_6(centerTemp.row(i), Column/3-1, clusterCenter.row(i));
				else if(normOption==7)
					tempMoving = getBMetric_7(centerTemp.row(i), Column/3-1, clusterCenter.row(i));
				else if(normOption==9)
					tempMoving = getBMetric_9(centerTemp.row(i), Column/3-1, clusterCenter.row(i));
				else
					tempMoving = getNorm(centerTemp.row(i),clusterCenter.row(i),normOption);*/
				// if previous and current centroids have same coordinates, then terminate
				tempMoving = (centerTemp.row(i)-clusterCenter.row(i)).norm(); 
				clusterCenter.row(i) = centerTemp.row(i);
				if(moving<tempMoving)
					moving = tempMoving;
			}
		}
		std::cout << "K-means iteration " << ++tag << " completed, and moving is " << moving 
				  << "!" << std::endl;
	}while(abs(moving-before)/before >= 1.0e-2 && tag <= 2 && moving > 1.0e-4);
	
	gettimeofday(&end, NULL);
	const double& delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;

	std::cout << "K-means takes " << delta << " s!" << std::endl;

	std::multimap<int,int> groupMap;
	entropy = 0.0;
	float probability;
	int increasingOrder[Cluster];
	for (int i = 0; i < Cluster; ++i)
	{
		groupMap.insert(std::pair<int,int>(storage[i],i));
		if(storage[i]>0)
		{
			probability = float(storage[i])/float(Row);
			entropy += probability*log(probability);
		}
	}
	entropy = -entropy;

	int groupNo = 0;
	for (std::multimap<int,int>::iterator it = groupMap.begin(); it != groupMap.end(); ++it)
	{
		if(it->first>0)
		{
			increasingOrder[it->second] = (groupNo++);
		}
	}
	/* finish tagging for each group */


	// set cluster group number and size number 
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int i = 0; i < Row; ++i)
	{
		group[i] = increasingOrder[recorder[i]];
		totalNum[i] = storage[recorder[i]];
	}

	float shortest, toCenter, farDist;
	int shortestIndex = 0, tempIndex = 0, furthestIndex = 0;
	std::vector<int> neighborTemp;

	const string& fileStr = string("../dataset/norm_")+to_string(normOption)+".txt";
	ofstream distFile(fileStr.c_str(),ios::out);
	if(!distFile)
	{
		std::cout << "Error created files!" << std::endl;
		exit(1);
	}

	/* choose cloest and furthest streamlines to centroid streamlines */
	for (int i = 0; i < Cluster; ++i)
	{
		if(storage[i]>0)
		{

			neighborTemp = neighborVec[i];
			shortest = FLT_MAX;
			farDist = FLT_MIN;

			for (int j = 0; j < storage[i]; ++j)
			{
				// j-th internal streamlines 
				tempIndex = neighborTemp[j];
				if(normOption==4)
					/* directly with mean rotation metric */
					toCenter = abs(rotation[tempIndex]-getRotation(clusterCenter.row(i), Column/3-2));
				else if(normOption==3)
					/* with B-metric of rotation sequences */
					toCenter = getBMetric_3(clusterCenter.row(i), Column/3-2, tempIndex, rotationSequence);
				else if(normOption==6)
					toCenter = getBMetric_6(clusterCenter.row(i), Column/3-1, tempIndex, normalMultivariate);
				else if(normOption==7)
					toCenter = getBMetric_7(clusterCenter.row(i), Column/3-1, tempIndex, rotationSequence);
				else if(normOption==9)
					toCenter = getBMetric_9(clusterCenter.row(i), Column/3-1, tempIndex, normalMultivariate);
				else
					toCenter = getNorm(clusterCenter.row(i),temp.row(tempIndex),normOption);
				distFile << toCenter << " ";

				/* update the closest index to centroid */
				if(toCenter<shortest)
				{
					shortest = toCenter;
					shortestIndex = tempIndex;
				}

				/* update the farthest index to centroid */
				if(toCenter>farDist)
				{
					farDist = toCenter;
					furthestIndex = tempIndex;
				}
			}
			closest.push_back(ExtractedLine(shortestIndex,increasingOrder[i]));
			furthest.push_back(ExtractedLine(furthestIndex,increasingOrder[i]));
			distFile << std::endl;
		}
	}
	distFile.close();

	std::cout << "Has taken closest and furthest out!" << std::endl;

	std::vector<float> closeSubset;

	/* based on known known cluster centroid, save them as vector for output */
	for (int i = 0; i < Cluster; ++i)
	{
		if(storage[i]>0)
		{
			for (int j = 0; j < Column; ++j)
			{
				closeSubset.push_back(clusterCenter(i,j));
			}
			massCenter.push_back(MeanLine(closeSubset,increasingOrder[i]));
			closeSubset.clear();
		}
	}
	delete[] storage;
}
