#pragma once
#include <vector>
#include "neuron.h"
class Network
{
public:
	Network(int n);
	~Network();
	void runDirectPropagation(float* inValues, int size);
	void runBackPropagation(float* inValues, int size);
	void countNetError(std::vector<float*> inValues, int size, int countError);
private:
	int era = 1;
	std::vector<Neuron> layers[3];
	void countNeuronValuesInLayer(float* inValues, std::vector<Neuron> &nextLayer, int size);
	float* getNewInputValues(std::vector<Neuron> layer);
	void countErrorForOutputLayer(float value1, float value2);
	std::vector<float> netAnswers[2];
	float nerErrors[2];
	std::vector<float> backError[3];
	void correctWeights(float* inValues, int layer, int size);
};

