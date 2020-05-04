#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <limits>
#include <math.h>
#include <iostream>

#include "network.h"

//#define attrCount 29
//#define classCount 3
//#define allCount (attrCount+classCount)
//#define totalCount (allCount-1)
//#define removed1 0
//#define removed2 19
//#define removed3 20


bool isFloat(std::string s) {
	std::istringstream iss(s);
	double dummy;
	iss >> std::noskipws >> dummy;
	return iss && iss.eof();     // Result converted to bool
}

int main(int argc, char* argv[]) {
	std::vector<std::string> title = std::vector<std::string>();
	std::vector<double*> values = std::vector<double*>();
	std::string stmp;
	double ftmp;
	std::ifstream in("dataset.txt");
	std::string titles;
	std::getline(in, titles);
	std::istringstream ss(titles);
	while (std::getline(ss, stmp, ';')) {
		if (stmp != "")
			title.push_back(stmp);
	}
	std::getline(in, stmp); // skip row with measuring units
	ss = std::istringstream(stmp);
	int n = 0;
	while (std::getline(ss, stmp, ';')) {
		n++;
	}
	n -= 2;
	int column = 0;
	int row = 0;
	while (std::getline(in, stmp)) { //line by line
		column = 0;
		ss = std::istringstream(stmp);
		std::getline(ss, stmp, ';');
		std::getline(ss, stmp, ';');
		values.push_back(new double[n]); // initialization new
		for (int i = 0; i < n; ++i)
			values.at(row)[i] = std::numeric_limits<double>::quiet_NaN();
		while (std::getline(ss, stmp, ';')) {	//delimeters ;
			if (isFloat(stmp)) {
				ftmp = std::stof(stmp);
				values.at(row)[column] = ftmp;
			}
			else
				values.at(row)[column] = std::numeric_limits<double>::quiet_NaN();
			column = ++column % n;
		}
		if (std::isnan(values.at(row)[n - 3])		//if all classes are null, remove
			&& std::isnan(values.at(row)[n - 2])
			&& std::isnan(values.at(row)[n - 1]))
			values.pop_back();
		else
			row++;
	}
	in.close();
	//union columns
	for (auto it = values.begin(); it != values.end(); ++it) {
		if (isnan((*it)[n - 2]) && !isnan((*it)[n - 1]))
			(*it)[n - 2] = (*it)[n - 1] * 1000;
	}
	double max1 = 0.0f;
	double max2 = 0.0f;
	for (int i = 0; i < values.size(); ++i) {
		if (values.at(i)[n - 3] > max1)
			max1 = values.at(i)[n - 3];
		if (values.at(i)[n - 2] > max2)
			max2 = values.at(i)[n - 2];
	}
	for (int i = 0; i < values.size(); ++i) {
		values.at(i)[n - 3] /= max1;
		values.at(i)[n - 2] /= max2;
	}
	//create network
	Network* net = new Network(n - 3);
	for (int i = 0; i < (int)(0.7*values.size()); i++) //test data is 70%
		net->runDirectPropagation(values.at(i), n - 3);
	net->countNetError(values, n - 1, (int)(0.7*values.size()));
	//std::ofstream out("out.txt");

	
	for (int era = 0; ; ++era)
	{
		for (int i = 0; i < (int)(0.7*values.size()); i++) {
			net->runBackPropagation(values.at(i), n - 1);
			net->runDirectPropagation(values.at(i), n - 1 - 2);
		}
		net->countNetError(values, n - 1, (int)(0.7*values.size()));
		std::cout << "era " << era << ": " << net->netErrors.at(era) << std::endl;
		//out << era << ";" << net->netErrors.at(era) << std::endl;
		if (net->netErrors.at(era) < 0.01)
			break;
	}
	std::cout << "era " << net->netErrors.size() - 1 << ": " << net->netErrors.at(net->netErrors.size() - 1) << std::endl;

	delete net;
	for (auto it = values.begin(); it != values.end(); ++it)
		delete[](*it);
	values.clear();
	title.clear();
	ss.clear();
	//out.close();
	system("pause");
	return 0;
}