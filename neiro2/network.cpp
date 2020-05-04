#include "network.h"
#define numNeuronsL1 50
#define numNeuronsL2 60
#define numNeuronsR 2
#define a 1
#define etta 0.001
#define fMin 0.04
#define fMax 0.06

#include<iostream>
#include <algorithm>
#include <ctime>
Network::Network(int size)
{
	layers[0] = std::vector<Neuron>();
	for (int i = 0; i < numNeuronsL1; ++i) {
		Neuron n = Neuron();
		n.id = i;
		n.sum = 0;
		std::vector<Link> links = std::vector<Link>();
		for (int j = 0; j < size; ++j) { //create links to n
			Link l = Link();
			l.idIn = j;
			l.idOut = i;
			srand(time(0));
			double f = (double)rand() / RAND_MAX;
			l.weight = 0;// fMin + f * (fMax - fMin);
			links.push_back(l);
		}
		n.inLinks = links;
		layers[0].push_back(n);
	}
	layers[1] = std::vector<Neuron>();
	for (int i = 0; i < numNeuronsL2; ++i) {
		Neuron n = Neuron();
		n.id = i;
		n.sum = 0;
		std::vector<Link> links = std::vector<Link>();
		for (int j = 0; j < numNeuronsL1; ++j) { //create links to n
			Link l = Link();
			l.idIn = j;
			l.idOut = i;
			srand(time(0));
			double f = (double)rand() / RAND_MAX;
			l.weight = 0;// fMin + f * (fMax - fMin);
			links.push_back(l);
		}
		n.inLinks = links;
		layers[1].push_back(n);
	}
	layers[2] = std::vector<Neuron>();
	for (int i = 0; i < numNeuronsR; ++i) {
		Neuron n = Neuron();
		n.id = i;
		n.sum = 0;
		std::vector<Link> links = std::vector<Link>();
		for (int j = 0; j < numNeuronsL2; ++j) { //create links to n
			Link l = Link();
			l.idIn = j;
			l.idOut = i;
			srand(time(0));
			double f = (double)rand() / RAND_MAX;
			l.weight = 0;// fMin + f * (fMax - fMin);
			links.push_back(l);
		}
		n.inLinks = links;
		layers[2].push_back(n);
	}
	netAnswers[0] = std::vector<double>();
	netAnswers[1] = std::vector<double>();
}


Network::~Network()
{
}

void Network::runDirectPropagation(double* inValues, int size) {
	for (int i = 0; i < 3; ++i) { //layers
		countNeuronValuesInLayer(inValues, layers[i], (i==0)?size:layers[i-1].size());
		inValues = getNewInputValues(layers[i]);
	}
	netAnswers[0].push_back(layers[2].at(0).activation);
	netAnswers[1].push_back(layers[2].at(1).activation);
	//std::cout << "1: " << netAnswers[0].at(netAnswers[0].size()-1) << std::endl << "2: " << netAnswers[1].at(netAnswers[1].size() - 1) <<std::endl;
}

void Network::countNeuronValuesInLayer(double* inValues, std::vector<Neuron> &nextLayer, int size) {
	for (int i = 0; i < nextLayer.size(); ++i) { //for each neurin from next layer
		double sum = 0;
		Neuron* neuron = &(nextLayer.at(i));
		for (int j = 0; j < size; ++j) { //count weightsum for each value from invalue
			double weight_j_i = neuron->inLinks.at(j).weight;
			double value = (std::isnan(inValues[j])) ? 0 : inValues[j];
			sum += (weight_j_i * value);
		}
		neuron->sum = sum;
		neuron->activation = 1.0 / (1 + exp(-neuron->sum));
	}
}


double * Network::getNewInputValues(std::vector<Neuron> layer)
{
	double* values = new double[layer.size()];
	for (int i = 0; i < layer.size(); ++i)
		values[i] = layer.at(i).activation;
	return values;
}

void Network::countNetError(std::vector<double*> inValues, int attrSize, int countExamples) {
	double error1 = 0;
	double error2 = 0;
	for (int i = 0; i < countExamples; ++i) {
		double value1 = isnan(inValues.at(i)[attrSize - 2])? 
			netAnswers[0][i] : 
			inValues.at(i)[attrSize - 2];
		double value2 = isnan(inValues.at(i)[attrSize - 1])? 
			netAnswers[1][i] : 
			inValues.at(i)[attrSize - 1];
		error1 += ((netAnswers[0].at(i) - value1)*(netAnswers[0].at(i) - value1));
		error2 += ((netAnswers[1].at(i) - value2)*(netAnswers[1].at(i) - value2));
	}
 	netErrors.push_back(std::max(error1, error2)/2);
	netAnswers[0].clear();
	netAnswers[1].clear();
}


void Network::runBackPropagation(double* inValues, int size) {
	countErrorForOutputLayer(inValues[size - 2], inValues[size - 1]);
	if (isnan(backError[2][0]) && isnan(backError[2][1]))
		return;
	for (int i = 1; i >= 0; --i) { //hidden layers
		for (int  k = 0; k < layers[i].size(); ++k) { //for each k-neuron from hidden layer
			double derivative = (layers[i].at(k).activation)*(1 - layers[i].at(k).activation);
			double sum = 0;
			for (int j = 0; j < layers[i+1].size(); ++j) { //for each neuron from next layer
				if (isnan(backError[i + 1][j]))
					continue;
				double weight = layers[i + 1].at(j).inLinks.at(k).weight;	//get weights which connecting with k-neuron
				sum += ((backError[i + 1].at(j) * weight) * derivative);	//weight multiply on sigma-error from next layer
			}
			backError[i].push_back(sum);
		}
	}
	for (int l = 0; l < 3; ++l)
		correctWeights(inValues, l, size-2);

	backError[0].clear();
	backError[1].clear();
	backError[2].clear();
	backError[0].shrink_to_fit();
	backError[1].shrink_to_fit();
	backError[2].shrink_to_fit();
}

void Network::correctWeights(double* inValues, int l, int size) {
	int prevN = (l == 0) ? size : layers[l - 1].size();
	std::vector<double> deltas(prevN, 0);
	for (int i = 0; i < layers[l].size(); ++i) { //index of next layer
		deltas = std::vector<double>(prevN, 0);
		for (int j = 0; j < prevN; ++j) {  //index prev
			if (isnan(backError[l][i])) continue;
			deltas.at(j) = -etta * (backError[l].at(i));
			if (l == 0)
				deltas.at(j) *= (isnan(inValues[j]) ? 0 : inValues[j]);
			else
				deltas.at(j) *= (layers[l - 1].at(j).activation);
			layers[l].at(i).inLinks.at(j).weight += deltas.at(j);
		}
		deltas.clear();
		deltas.shrink_to_fit();
	}
}

void Network::countErrorForOutputLayer(double value1, double value2) {
	double activation1 = layers[2].at(0).activation;
	double activation2 = layers[2].at(1).activation;
	backError[2].push_back((activation1 - value1)*activation1*(1 - activation1));
	backError[2].push_back((activation2 - value2)*activation2*(1 - activation2));
}