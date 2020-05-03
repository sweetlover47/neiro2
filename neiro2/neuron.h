#pragma once
#include <vector>
#include "link.h"
class Neuron
{
public:
	Neuron();
	~Neuron();
	int id;
	float sum;
	std::vector<Link> inLinks; // -> (N)
	float activation;
private:
	
};

