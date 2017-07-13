#include "IOHandler.h"


void IOHandler::readFile(const string& fileName, 
						 std::vector< std::vector<float > >& dataVec, 
						 int& vertexCount, 
						 const int& dimension)
{
	vertexCount = 0;
	std::ifstream fin(fileName.c_str(), ios::in);
	if(!fin)
	{
		std::cout << "Error creating files!" << std::endl;
		exit(1);
	}
	stringstream ss;
	std::vector<float> tempVec;

	string line, part;
	while(getline(fin, line))
	{
		ss.str(line);
		/* extract point coordinates from data file */
		while(ss >> part)
		{
			tempVec.push_back(atof(part.c_str()));
		}
		/* accept only streamlines with at least three vertices */
		if(tempVec.size()/3>3)
		{
			dataVec.push_back(tempVec);
			vertexCount+=tempVec.size();
		}
		tempVec.clear();
		ss.clear();
		ss.str("");
	}
	fin.close();

	vertexCount/=dimension;
	std::cout << "File reader has been completed, and it toally has " << dataVec.size() << " trajectories and " 
	          << vertexCount << " vertices!" << std::endl;
}


void IOHandler::readFile(const string& fileName, 
						 std::vector< std::vector<float > >& dataVec, 
						 int& vertexCount, 
						 const int& dimension,
						 const int& trajectoryNum, 
						 const int& Frame)
{
	vertexCount = trajectoryNum*(Frame-1);
	dataVec = std::vector< std::vector<float> >(trajectoryNum, std::vector<float> ((Frame-1)*dimension));
#pragma omp parallel for schedule(dynamic) num_threads(8)
	/* from 1 to Frame-1 then pay attention to i index */
	for (int i = 1; i < Frame; ++i)
	{
		stringstream ss;
		ss << fileName << i << ".txt";
		std::ifstream fin(ss.str().c_str(), ios::in);
		if(!fin)
		{
			std::cout << "File doesn't exist for this number!" << std::endl;
			exit(1);
		}
		float firstFloat;
		string line, linePart;

		ss.clear();
		ss.str("");
		for (int j = 0; j < trajectoryNum; ++j)
		{
			getline(fin, line);

			assert(!line.empty());

			ss.str(line);
			ss >> linePart;

			ss >> linePart;
			dataVec[j][(i-1)*dimension] = atof(linePart.c_str());

			ss >> linePart;
			dataVec[j][(i-1)*dimension+1] = atof(linePart.c_str());

			ss >> linePart;
			dataVec[j][(i-1)*dimension+2] = atof(linePart.c_str());
		}

		fin.close();
		std::cout << "File " << i << " has been read in successfully!" << std::endl;
	}


}


void IOHandler::printVTK(const string& fileName, 
						 const std::vector< std::vector<float > >& dataVec, 
						 const int& vertexCount, 
						 const int& dimension,
						 const std::vector<int>& clusterNumber)
{
	std::ofstream fout(fileName.c_str(), ios::out);
	if(!fout)
	{
		std::cout << "Error creating a new file!" << std::endl;
		exit(1);
	}
	fout << "# vtk DataFile Version 3.0" << std::endl << "Bernard streamline" << std::endl
	     << "ASCII" << std::endl << "DATASET POLYDATA" << std::endl;
	fout << "POINTS " << vertexCount << " float" << std::endl;

	int subSize, arraySize;
	std::vector<float> tempVec;
	for (int i = 0; i < dataVec.size(); ++i)
	{
		tempVec = dataVec[i];
		subSize = tempVec.size()/dimension;
		for (int j = 0; j < subSize; ++j)
		{
			for (int k = 0; k < dimension; ++k)
			{
				fout << tempVec[j*dimension+k] << " ";
			}
			fout << endl;
		}
	}

	fout << "LINES " << dataVec.size() << " " << (vertexCount+dataVec.size()) << std::endl;

	subSize = 0;
	for (int i = 0; i < dataVec.size(); ++i)
	{
		arraySize = dataVec[i].size()/dimension;
		fout << arraySize << " ";
		for (int j = 0; j < arraySize; ++j)
		{
			fout << subSize+j << " ";
		}
		subSize+=arraySize;
		fout << std::endl;
	}
	fout << "POINT_DATA" << " " << vertexCount << std::endl;
	fout << "SCALARS group int 1" << std::endl;
	fout << "LOOKUP_TABLE group_table" << std::endl;

	for (int i = 0; i < dataVec.size(); ++i)
	{
		arraySize = dataVec[i].size()/dimension;
		for (int j = 0; j < arraySize; ++j)
		{
			fout << clusterNumber[i] << std::endl;
		}
	}

	fout.close();
}


void IOHandler::printVTK(const string& fileName, 
						 const std::vector< std::vector<float > >& dataVec, 
						 const int& vertexCount, 
						 const int& dimension)
{
	std::ofstream fout(fileName.c_str(), ios::out);
	if(!fout)
	{
		std::cout << "Error creating a new file!" << std::endl;
		exit(1);
	}
	fout << "# vtk DataFile Version 3.0" << std::endl << "Bernard streamline" << std::endl
	     << "ASCII" << std::endl << "DATASET POLYDATA" << std::endl;
	fout << "POINTS " << vertexCount << " float" << std::endl;

	int subSize, arraySize;
	std::vector<float> tempVec;
	for (int i = 0; i < dataVec.size(); ++i)
	{
		tempVec = dataVec[i];
		subSize = tempVec.size()/dimension;
		for (int j = 0; j < subSize; ++j)
		{
			for (int k = 0; k < dimension; ++k)
			{
				fout << tempVec[j*dimension+k] << " ";
			}
			fout << endl;
		}
	}

	fout << "LINES " << dataVec.size() << " " << (vertexCount+dataVec.size()) << std::endl;

	subSize = 0;
	for (int i = 0; i < dataVec.size(); ++i)
	{
		arraySize = dataVec[i].size()/dimension;
		fout << arraySize << " ";
		for (int j = 0; j < arraySize; ++j)
		{
			fout << subSize+j << " ";
		}
		subSize+=arraySize;
		fout << std::endl;
	}
	fout << "POINT_DATA" << " " << vertexCount << std::endl;
	fout << "SCALARS group int 1" << std::endl;
	fout << "LOOKUP_TABLE group_table" << std::endl;

	for (int i = 0; i < dataVec.size(); ++i)
	{
		arraySize = dataVec[i].size()/dimension;
		for (int j = 0; j < arraySize; ++j)
		{
			fout << i << std::endl;
		}
	}

	fout.close();
}


void IOHandler::printVTK(const string& fileName, 
						 const std::vector<MeanLine>& dataVec, 
						 const int& vertexCount, 
						 const int& dimension)
{
	std::ofstream fout(fileName.c_str(), ios::out);
	if(!fout)
	{
		std::cout << "Error creating a new file!" << std::endl;
		exit(1);
	}
	fout << "# vtk DataFile Version 3.0" << std::endl << "Bernard streamline" << std::endl
	     << "ASCII" << std::endl << "DATASET POLYDATA" << std::endl;
	fout << "POINTS " << vertexCount << " float" << std::endl;

	int subSize, arraySize;
	std::vector<float> tempVec;
	for (int i = 0; i < dataVec.size(); ++i)
	{
		tempVec = dataVec[i].minCenter;
		subSize = tempVec.size()/dimension;
		for (int j = 0; j < subSize; ++j)
		{
			for (int k = 0; k < dimension; ++k)
			{
				fout << tempVec[j*dimension+k] << " ";
			}
			fout << endl;
		}
	}

	fout << "LINES " << dataVec.size() << " " << (vertexCount+dataVec.size()) << std::endl;

	subSize = 0;
	for (int i = 0; i < dataVec.size(); ++i)
	{
		arraySize = dataVec[i].minCenter.size()/dimension;
		fout << arraySize << " ";
		for (int j = 0; j < arraySize; ++j)
		{
			fout << subSize+j << " ";
		}
		subSize+=arraySize;
		fout << std::endl;
	}
	fout << "POINT_DATA" << " " << vertexCount << std::endl;
	fout << "SCALARS group int 1" << std::endl;
	fout << "LOOKUP_TABLE group_table" << std::endl;

	for (int i = 0; i < dataVec.size(); ++i)
	{
		arraySize = dataVec[i].minCenter.size()/dimension;
		for (int j = 0; j < arraySize; ++j)
		{
			fout << dataVec[i].cluster << std::endl;
		}
	}

	fout.close();
}



void IOHandler::expandArray(float ***data, 
							const std::vector< std::vector<float> >& dataVec, 
							const int& dimension, 
							int& maxElements)
{
	maxElements = INT_MIN;
	int arraySize;
	for (int i = 0; i < dataVec.size(); ++i)
	{
		arraySize = dataVec[i].size();
		if(maxElements < arraySize)
			maxElements = arraySize;
	}
	std::cout << maxElements << std::endl;

	*data = new float*[dataVec.size()];
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int i = 0; i < dataVec.size(); ++i)
	{
		const int& vecSize = dataVec[i].size();
		(*data)[i] = new float[maxElements];
		memcpy(&(*data)[i][0], &(dataVec[i][0]), vecSize*sizeof(float));
		for (int j = vecSize; j < maxElements; j=j+dimension)
		{
			memcpy(&(*data)[i][j], &(*data)[i][vecSize-dimension], dimension*sizeof(float));
		}
	}
}


void IOHandler::formArray(float ***data, 
						  const std::vector< std::vector<float> >& dataVec, 
						  const int& dimension)
{
	*data = new float*[dataVec.size()];
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int i = 0; i < dataVec.size(); ++i)
	{
		const int& arraySize = dataVec[i].size();
		(*data)[i] = new float[arraySize];
		memcpy(&(*data)[i][0], &(dataVec[i][0]), arraySize*sizeof(float));
	}
}


void IOHandler::printToFull(const std::vector< std::vector<float> >& dataVec, 
							const std::vector<int>& group, 
				 			const std::vector<int>& totalNum, 
				 			const string& groupName, 
				 			const string& fullName, 
				 			const int& dimension)
{
	std::ofstream fout(fullName.c_str(), ios::out | ios::app );
	if(!fout)
	{
		std::cout << "Error opening the file!" << std::endl;
		exit(1);
	}

	fout << "SCALARS " << groupName << " int 1" << std::endl;
	fout << "LOOKUP_TABLE " << groupName+string("_table") << std::endl;

	int arraySize;
	for (int i = 0; i < dataVec.size(); ++i)
	{
		arraySize = dataVec[i].size()/dimension;
		for (int j = 0; j < arraySize; ++j)
		{
			fout << group[i] << std::endl;
		}
	}

	fout << "SCALARS " <<  groupName+string("_num") << " int 1" << std::endl;
	fout << "LOOKUP_TABLE " <<  groupName+string("_num_table") << std::endl;

	for (int i = 0; i < dataVec.size(); ++i)
	{
		arraySize = dataVec[i].size()/dimension;
		for (int j = 0; j < arraySize; ++j)
		{
			fout << totalNum[i] << std::endl;
		}
	}
	fout.close(); 
}


void IOHandler::deleteArray(float **data, 
							const int& row)
{
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int i = 0; i < row; ++i)
	{
		delete[] data[i];
	}
	delete[] data;
}


void IOHandler::writeReadme(const double& PCA_KMeans_delta, 
							const double& KMeans_delta)
{
	std::ofstream readme("../dataset/README",ios::out | ios::app);
	if(!readme)
	{
		std::cout << "Error creating readme!" << std::endl;
		exit(1);
	}
	readme << "PCA_KMeans time elapse is " << PCA_KMeans_delta << " s." << std::endl
		   << "KMeans time elapse is " << KMeans_delta << " s." << std::endl;
    readme << std::endl;
    readme.close();
}


void IOHandler::writeReadme(const std::vector<string>& timeName, 
							const std::vector<double>& timeDiff,
							const int& cluster,
							const std::vector<float>& entropyVec)
{
	std::ofstream readme("../dataset/README",ios::out | ios::app);
	if(!readme)
	{
		std::cout << "Error creating readme!" << std::endl;
		exit(1);
	}
	assert(timeName.size()==timeDiff.size());

	for (int i = 0; i < timeName.size(); ++i)
	{
		readme << timeName[i] << " is " << timeDiff[i] << " s." << std::endl;
	}
	readme << std::endl;
	readme << "Preset cluster number in K-means is: " << cluster << std::endl;
	readme << std::endl;
	readme << "Entropy of each norm-based K-means is: " << std::endl;
	for (int i = 0; i < entropyVec.size(); ++i)
	{
		readme << entropyVec[i] << " ";
	}
    readme.close();
}

void IOHandler::writeReadme(const std::vector<ExtractedLine>& closest, 
							const std::vector<ExtractedLine>& furthest, 
							const int& normOption)
{
	std::ofstream readme("../dataset/README",ios::out | ios::app);
	if(!readme)
	{
		std::cout << "Error creating readme!" << std::endl;
		exit(1);
	}
	const string& normStr = "Norm_"+to_string(normOption);
	readme << std::endl;
	readme << normStr+ " closest streamline set has " << closest.size() << " streamlines" << std::endl;
	for (int i = 0; i < closest.size(); ++i)
	{
		readme << closest[i].lineNum << " ";
	}
	readme << std::endl;
	
	readme << std::endl;
	readme << normStr+ " furthest streamline set has " << furthest.size() << " streamlines" << std::endl;
	for (int i = 0; i < furthest.size(); ++i)
	{
		readme << furthest[i].lineNum << " ";
	}
	readme << std::endl;
    readme.close();
}


void IOHandler::writeReadme(const std::vector<ExtractedLine>& closest, 
							const std::vector<ExtractedLine>& furthest)
{
	std::ofstream readme("../dataset/README",ios::out | ios::app);
	if(!readme)
	{
		std::cout << "Error creating readme!" << std::endl;
		exit(1);
	}
	readme << std::endl;
	readme << "PCA closest streamline set has " << closest.size() << " streamlines" << std::endl;
	for (int i = 0; i < closest.size(); ++i)
	{
		readme << closest[i].lineNum << " ";
	}
	readme << std::endl;
	
	readme << std::endl;
	readme << "PCA furthest streamline set has " << furthest.size() << " streamlines" << std::endl;
	for (int i = 0; i < furthest.size(); ++i)
	{
		readme << furthest[i].lineNum << " ";
	}
	readme << std::endl;
    readme.close();
}


void IOHandler::assignVec(std::vector<std::vector<float> >& closestStreamline, 
						  std::vector<int>& cluster,
						  const std::vector<ExtractedLine>& closest,
						  int& pointNumber, 
						  const std::vector< std::vector<float> >& dataVec)
{
	closestStreamline = std::vector<std::vector<float> >(closest.size(), std::vector<float>());
	cluster = std::vector<int>(closest.size());
	pointNumber = 0;
	for (int i = 0; i < closestStreamline.size(); ++i)
	{
		closestStreamline[i] = dataVec[closest[i].lineNum];
		pointNumber+=closestStreamline[i].size();
		cluster[i] = closest[i].cluster;
	}	
}


void IOHandler::assignVec(std::vector<int>& cluster,
						  const std::vector<MeanLine>& centerMass)
{
	cluster = std::vector<int>(centerMass.size());
#pragma omp parallel for schedule(dynamic) num_threads(8)
	for (int i = 0; i < cluster.size(); ++i)
	{
		cluster[i] = centerMass[i].cluster;
	}
}


void IOHandler::writeGroup(const std::vector<int>& group, 
						   const std::vector< std::vector<float> >& dataVec)
{
	std::ofstream readme("../dataset/group",ios::out);
	if(!readme)
	{
		std::cout << "Error creating readme!" << std::endl;
		exit(1);
	}
	assert(group.size()==dataVec.size());
	for (int i = 0; i < group.size(); ++i)
	{
		readme << group[i] << std::endl;
	}
    readme.close();
}