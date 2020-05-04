#pragma once
#include <vector>
#include "neuron.h"
class Network
{
public:
	Network(int n);
	~Network();
	void runDirectPropagation(double* inValues, int size);
	void runBackPropagation(double* inValues, int size);
	void countNetError(std::vector<double*> inValues, int size, int countError);
	std::vector<double> netErrors;

private:
	int era = 1;
	std::vector<Neuron> layers[3];
	void countNeuronValuesInLayer(double* inValues, std::vector<Neuron> &nextLayer, int size);
	double* getNewInputValues(std::vector<Neuron> layer);
	void countErrorForOutputLayer(double value1, double value2);
	std::vector<double> netAnswers[2];
	std::vector<double> backError[3];
	void correctWeights(double* inValues, int layer, int size);
};

