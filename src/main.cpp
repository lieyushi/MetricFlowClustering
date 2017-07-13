#include "PCA_Cluster.h"
#include <sys/time.h>

using namespace std;


void performPCA_Cluster(const string& fileName, 
						const std::vector< std::vector<float> >& dataVec, 
						const int& cluster, 
						const int& dimension, 
						const string& fullName, 
						const int& maxElements, 
						float **data,
						float& entropy);

void performK_Means(const string& fileName, 
					const std::vector< std::vector<float> >& dataVec, 
					const int& cluster, 
					const int& dimension, 
					const string& fullName, 
					const int& maxElements, 
					float **data,
					const int& normOption,
					float& entropy);

int initializationOption;

int main(int argc, char* argv[])
{
	while(argc!=3)
	{
		std::cout << "Input argument should have 3!" << endl
		          << "./cluster inputFile data_dimension" << endl;
		exit(1);
	}
	const string& strName = string("../dataset/")+string(argv[1]);
	const int& dimension = atoi(argv[2]);
	
	int cluster, vertexCount;
	std::cout << "Please input a cluster number (>=2):" << std::endl;
	std::cin >> cluster;

	std::cout << "Please choose initialization option for seeds:" << std::endl
			  << "1.chose random positions, 2.Chose from samples, 3.k-means++ sampling" << endl;
	std::cin >> initializationOption;
	assert(initializationOption==1 || initializationOption==2 || initializationOption==3);

	std::vector<string> timeName;
	std::vector<double> timeDiff;
	std::vector<float> entropyVec;
	float entropy;

	struct timeval start, end;
	double timeTemp;

	gettimeofday(&start, NULL);
	std::vector< std::vector<float> > dataVec;
	IOHandler::readFile(strName, dataVec, vertexCount, dimension);
	//IOHandler::readFile(pbfPath, dataVec, vertexCount, dimension, 128000, 1500);
	gettimeofday(&end, NULL);
	timeTemp = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
	timeName.push_back("I-O file reader time");
	timeDiff.push_back(timeTemp);

	stringstream ss;
	ss << strName << "_differentNorm_full.vtk";
	const string& fullName = ss.str();
	IOHandler::printVTK(ss.str(), dataVec, vertexCount, dimension);
	ss.str("");

	int maxElements;
	float **data = NULL, **originData = NULL;
	IOHandler::expandArray(&data, dataVec, dimension, maxElements);

	ss << strName << "_PCAClustering";
	gettimeofday(&start, NULL);
	//ss << "TACs_PCAClustering";
	performPCA_Cluster(ss.str(), dataVec, cluster, dimension, 
					   fullName, maxElements, data, entropy);
	entropyVec.push_back(entropy);
	ss.str("");
	ss.clear();
	gettimeofday(&end, NULL);
	timeTemp = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
	timeName.push_back("PCA+K_Means operation time");
	timeDiff.push_back(timeTemp);

	/*  0: Euclidean Norm
		1: Fraction Distance Metric
		2: piece-wise angle average
		3: Bhattacharyya metric for rotation
		4: average rotation
		5: signed-angle intersection
		6: normal-direction multivariate distribution
		7: Bhattacharyya metric with angle to a fixed direction
		8: Piece-wise angle average \times standard deviation
		9: normal-direction multivariate un-normalized distribution
		10: acos(x*y/x.size) which measures the average angles in machine learning
	*/

	for(int i = 0;i<11;i++)
	{
		gettimeofday(&start, NULL);
		ss << strName << "_KMeans";
		//ss << "TACs_KMeans";
		performK_Means(ss.str(), dataVec, cluster, dimension, 
					   fullName, maxElements, data,i, entropy);
		entropyVec.push_back(entropy);
		ss.str("");
		gettimeofday(&end, NULL);
		timeTemp = ((end.tv_sec  - start.tv_sec) * 1000000u 
					+ end.tv_usec - start.tv_usec) / 1.e6;
		timeName.push_back("Direct K_Means operation time for norm "+to_string(i));
		timeDiff.push_back(timeTemp);
	}

	IOHandler::deleteArray(data, dataVec.size());

	IOHandler::writeReadme(timeName, timeDiff, cluster, entropyVec);

	return 0;
}


void performPCA_Cluster(const string& fileName, 
					    const std::vector< std::vector<float> >& dataVec, 
					    const int& cluster, 
						const int& dimension, 
						const string& fullName, 
						const int& maxElements, 
						float **data,
						float& entropy)
{

	std::vector<MeanLine> centerMass;
	std::vector<int> group(dataVec.size());
	std::vector<ExtractedLine> closest;
	std::vector<ExtractedLine> furthest;
	std::vector<int> totalNum(dataVec.size());
	PCA_Cluster::performPCA_Clustering(data, dataVec.size(), maxElements, 
						centerMass, group, totalNum, closest, furthest, cluster, entropy);
	std::vector<std::vector<float> > closestStreamline;
	std::vector<std::vector<float> > furthestStreamline;
	std::vector<int> closestCluster, furthestCluster, meanCluster;
	int closestPoint, furthestPoint;
	IOHandler::assignVec(closestStreamline, closestCluster, closest, closestPoint, dataVec);
	IOHandler::assignVec(furthestStreamline, furthestCluster, furthest, furthestPoint, dataVec);
	IOHandler::assignVec(meanCluster, centerMass);
	IOHandler::printVTK(fileName+string("_PCA_closest.vtk"), closestStreamline, 
						closestPoint/dimension, dimension, closestCluster);
	IOHandler::printVTK(fileName+string("_PCA_furthest.vtk"), furthestStreamline, 
						furthestPoint/dimension, dimension, furthestCluster);
	IOHandler::printVTK(fileName+string("_PCA_mean.vtk"), centerMass, 
						centerMass.size()*centerMass[0].minCenter.size()/dimension, 
						dimension);
	std::cout << "Finish printing vtk for pca-clustering result!" << std::endl;

	IOHandler::printToFull(dataVec, group, totalNum, string("PCA_KMeans"), fullName, dimension);
	IOHandler::writeReadme(closest, furthest);
	//IOHandler::writeGroup(group, dataVec);
}


void performK_Means(const string& fileName, 
					const std::vector< std::vector<float> >& dataVec, 
					const int& cluster, 
					const int& dimension, 
					const string& fullName, 
					const int& maxElements, 
					float **data, 
					const int& normOption,
					float& entropy)
{
	std::vector<MeanLine> centerMass;
	std::vector<ExtractedLine> closest;
	std::vector<ExtractedLine> furthest;
	std::vector<int> group(dataVec.size());
	std::vector<int> totalNum(dataVec.size());
	PCA_Cluster::performDirectK_Means(data, dataVec.size(), maxElements, 
									  centerMass, group, totalNum, 
									  closest, furthest, cluster, normOption, entropy);
	std::vector<std::vector<float> > closestStreamline, furthestStreamline;
	std::vector<int> closestCluster, furthestCluster, meanCluster;
	int closestPoint, furthestPoint;
	IOHandler::assignVec(closestStreamline, closestCluster, closest, closestPoint, dataVec);
	IOHandler::assignVec(furthestStreamline, furthestCluster, furthest, furthestPoint, dataVec);
	IOHandler::assignVec(meanCluster, centerMass);
	IOHandler::printVTK(fileName+string("_norm")+to_string(normOption)+string("_mean.vtk"), 
						centerMass, centerMass.size()*centerMass[0].minCenter.size()/dimension, dimension);
	IOHandler::printVTK(fileName+string("_norm")+to_string(normOption)+string("_closest.vtk"), 
						closestStreamline, closestPoint/dimension, dimension, closestCluster);
	IOHandler::printVTK(fileName+string("_norm")+to_string(normOption)+string("_furthest.vtk"), 
						furthestStreamline, furthestPoint/dimension, dimension, furthestCluster);
	std::cout << "Finish printing vtk for k-means clustering result!" << std::endl;

	IOHandler::printToFull(dataVec, group, totalNum, string("norm")+to_string(normOption)
						   +string("_KMeans"), fullName, dimension);
	IOHandler::writeReadme(closest, furthest, normOption);
	centerMass.clear();
	closest.clear();
	furthest.clear();
	group.clear();
	totalNum.clear();
}


