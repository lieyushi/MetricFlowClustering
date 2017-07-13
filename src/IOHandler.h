#ifndef _IOHANDLER_H_
#define _IOHANDLER_H_

#include <fstream>
#include <vector>
#include <iostream>
#include <cstring>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <climits>
#include <cassert>

using namespace std;

struct ExtractedLine
{
	int lineNum;
	int cluster;
	ExtractedLine(const int& pointIndex,
				  const int& cluster)
				 :lineNum(pointIndex),cluster(cluster)
	{}
};


struct MeanLine
{
	std::vector<float> minCenter;
	int cluster;
	MeanLine(const std::vector<float>& minCenter,
			 const int& cluster)
	        :minCenter(minCenter), cluster(cluster)
	{}
};

class IOHandler
{
	
public:
	
	static void readFile(const string& fileName, 
						 std::vector< std::vector<float > >& dataVec, 
						 int& vertexCount, 
						 const int& dimension);

	static void readFile(const string& fileName, 
						 std::vector< std::vector<float > >& dataVec, 
						 int& vertexCount, 
						 const int& dimension,
						 const int& trajectoryNum, 
						 const int& Frame);

	static void printVTK(const string& fileName, 
						 const std::vector< std::vector<float > >& dataVec, 
						 const int& vertexCount, 
						 const int& dimension,
						 const std::vector<int>& clusterNumber);

	static void printVTK(const string& fileName, 
						 const std::vector< std::vector<float > >& dataVec, 
						 const int& vertexCount, 
						 const int& dimension);

	static void printVTK(const string& fileName, 
						 const std::vector<MeanLine>& dataVec, 
						 const int& vertexCount, 
						 const int& dimension);

	static void printToFull(const std::vector< std::vector<float> >& dataVec, 
							const std::vector<int>& group, 
				 			const std::vector<int>& totalNum, 
				 			const string& groupName, 
				 			const string& fullName, 
				 			const int& dimension);


	static void writeReadme(const double& PCA_KMeans_delta, 
							const double& KMeans_delta);

	static void writeReadme(const std::vector<string>& timeName, 
							const std::vector<double>& timeDiff,
							const int& cluster,
							const std::vector<float>& entropyVec);

	static void writeReadme(const std::vector<ExtractedLine>& closest, 
							const std::vector<ExtractedLine>& furthest, 
							const int& normOption);

	static void writeReadme(const std::vector<ExtractedLine>& closest, 
							const std::vector<ExtractedLine>& furthest);

/* expand array to the longest size so that we can perform entrywise comparison */
	static void expandArray(float ***data, 
							const std::vector< std::vector<float> >& dataVec, 
							const int& dimension, 
							int& maxElements);

/* form array directly copied by vector for distribution-based comparison */
	static void formArray(float ***data, 
						  const std::vector< std::vector<float> >& dataVec, 
						  const int& dimension);

	static void deleteArray(float **data, 
							const int& row);

	static void assignVec(std::vector<std::vector<float> >& closestStreamline,
						  std::vector<int>& cluster, 
						  const std::vector<ExtractedLine>& closest, 
						  int& pointNumber,
						  const std::vector< std::vector<float> >& dataVec);

	static void assignVec(std::vector<int>& cluster,
						  const std::vector<MeanLine>& centerMass);

	static void writeGroup(const std::vector<int>& group, 
						   const std::vector< std::vector<float> >& dataVec);

};


#endif