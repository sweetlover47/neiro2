#pragma once
#include <vector>
#include "link.h"
class Neuron
{
public:
	Neuron();
	~Neuron();
	int id;
	double sum;
	std::vector<Link> inLinks; // -> (N)
	double activation;
private:
	
};

