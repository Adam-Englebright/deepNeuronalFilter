#include "Net.h"
#include "Layer.h"
#include "Neuron.h"

#include <cstddef>
#include <mutex>
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <ctgmath>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <fstream>
#include <iostream>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string>
#include <numeric>
#include <vector>
#include <vector>

using namespace std;

//*************************************************************************************
//initialisation:
//*************************************************************************************

Net::Net(const int _nLayers, const int *const _nNeurons, const int _nInputs,
         boost::circular_buffer<double> &_inputBuffer, const int _subject,
         const string _trial, const unsigned char _nThreads)
	:
	inputBuffer(_inputBuffer) {
	nLayers = _nLayers; //no. of layers including inputs and outputs layers
	layers= new Layer*[(unsigned)nLayers];
	const int* nNeuronsp = _nNeurons; //number of neurons in each layer
	nInputs=_nInputs; // the no. of inputs to the network (i.e. the first layer)
	//cout << "nInputs: " << nInputs << endl;
	int nInput = 0; //temporary variable to use within the scope of for loop
	for (int i=0; i<nLayers; i++){
		int numNeurons= *nNeuronsp; //no. neurons in this layer
		if (i==0){nInput=nInputs;}
		/* no. inputs to the first layer is equal to no. inputs to the network */
		layers[i]= new Layer(numNeurons, nInput, _subject, _trial);
		nNeurons += numNeurons;
		nWeights += (numNeurons * nInput);
		nInput=numNeurons;
		/*no. inputs to the next layer is equal to the number of neurons
		 * in the current layer. */
		nNeuronsp++; //point to the no. of neurons in the next layer
	}
	nOutputs=layers[nLayers-1]->getnNeurons();
	errorGradient= new double[(unsigned)nLayers];
	//cout << "net" << endl;

	// Use network data to distrubute neurons on each layer for forward propagation threads
	// to work on. Load thread meta data structs with this data and start threads.
	if (_nThreads > 1) {
		// Create vector of ForwardPropThreadMetaData structs, one for each thread.
		std::vector<ThreadMetaData> threadMetaDataVec;
		for (size_t i=0; i<_nThreads; ++i) {
			ThreadMetaData metaData;
			threadMetaDataVec.push_back(metaData);
		}

		// For each layer, add neuron indices for processing to
		// the meta data for each thread.
		for (int i=0; i<nLayers; ++i) {
			size_t neuronsPerLayer = layers[i]->getnNeurons();
			size_t threadsPerLayerCount = 0;

			// Vector storing vectors of neuron indices for each thread,
			// for each layer.
			std::vector<std::vector<size_t>> neuronIndexVecVec;
			for (size_t j=0; j<_nThreads; ++j) {
				std::vector<size_t> neuronIndexVec;
				neuronIndexVecVec.push_back(neuronIndexVec);
			}

			for (size_t j=0; j<neuronsPerLayer; ++j) {
				neuronIndexVecVec[j % _nThreads].push_back(j);
			}

			for (size_t j=0; j<_nThreads; ++j) {
				threadMetaDataVec[j].neuronIndexVecVec.push_back(neuronIndexVecVec[j]);
				if (neuronIndexVecVec[j].size() > 0) {
					threadMetaDataVec[j].numLayers++;
					threadsPerLayerCount++;
				}
			}

			// Add threads per layer data to thread meta data structs.
			for (size_t j=0; j<_nThreads; ++j) {
				threadMetaDataVec[j].numThreadsVec.push_back(threadsPerLayerCount);
			}
		}

		// Get number of threads actually doing anything. Must be done before starting threads.
		for (auto metaData : threadMetaDataVec) {
			if (metaData.numLayers > 0) {
				noThreadsWorking++;
			}
		}

		// Spin up the forward propagation worker threads.
		for (auto metaData : threadMetaDataVec) {
			if (metaData.numLayers > 0) {
				filterThreadPool.push_back(std::thread(&Net::filterThread, this, metaData));
			}
		}
	}
}

Net::~Net(){
	// Have threads terminate.
	filterThreadsTerm = true;
	{
		std::lock_guard<std::mutex> lk(filterStartMtx);
		filterStartCond = true;
	}
	filterStartCV.notify_all();

	for (auto& thread : filterThreadPool) {
		thread.join();
	}
	
	for (int i=0; i<nLayers; i++){
		delete layers[i];
	}
	delete[] layers;
	delete[] errorGradient;
}

void Net::initNetwork(Neuron::weightInitMethod _wim, Neuron::biasInitMethod _bim, Neuron::actMethod _am){
	for (int i=0; i<nLayers; i++){
		layers[i]->initLayer(i, _wim, _bim, _am);
	}
}

void Net::setLearningRate(double _w_learningRate, double _b_learningRate){
	for (int i=0; i<nLayers; i++){
		layers[i]->setlearningRate(_w_learningRate, _b_learningRate);
	}
}

//*************************************************************************************
//forward propagation of inputs:
//*************************************************************************************

void Net::setInputs(const boost::circular_buffer<double>& _inputs, const double scale, const unsigned int offset, const int n) {
	layers[0]->setInputs(_inputs, scale, offset, n); //sets the inputs to the first layer only
}

void Net::propInputs(){
	for (int i=0; i<nLayers-1; i++){
		layers[i]->calcOutputs();
		for (int j=0; j<layers[i]->getnNeurons(); j++){
			double inputOuput = layers[i]->getOutput(j);
			layers[i+1]->propInputs(j, inputOuput);
		}
	}
	layers[nLayers-1]->calcOutputs();
	/* this calculates the final outoup of the network,
	 * i.e. the output of the final layer
	 * but this is not fed into any further layer*/
}

double Net::filterMT(double _delayed_signal) {
	delayed_signal = _delayed_signal;
	
	// Busy loop until all threads are ready to go.
	while (filterThreadReadyCount < noThreadsWorking) {}

	// Reset the value of networkComponentFinishedCount and filterThreadReadyCount
	networkComponentFinishedCount.store(0);
	filterThreadReadyCount.store(0);
	
	// Send notification to forward propagation worker threads to start.
	{
		std::lock_guard<std::mutex> lk(filterStartMtx);
		filterStartCond = true;
	}
	filterStartCV.notify_all();

	// Now wait until forward propagation worker threads have finished.
	{
		std::unique_lock<std::mutex> lk(filterFinishedMtx);
		filterFinishedCV.wait(lk, [this]{ return filterFinishedCond; });
		filterFinishedCond = false;
	}

	// Return the filter result
	return f_nn;
}

void Net::filterThread(ThreadMetaData metaData) {
	while (true) {
		size_t networkComponentCount = 0;

		// Wait for notification to start from main thread.
		{
			std::unique_lock<std::mutex> lk(filterStartMtx);
			filterStartCond = false;
			filterThreadReadyCount++;
			filterStartCV.wait(lk, [this]{ return filterStartCond; });
		}

		// Check if the thread is to terminate.
		if (filterThreadsTerm)
			break;

		// Set inputs to first layer.
		networkComponentCount += noThreadsWorking;
		
		layers[0]->setInputsVec(inputBuffer, metaData.neuronIndexVecVec[0]);
		
		networkComponentFinishedCount++;
		while (networkComponentFinishedCount.load() < networkComponentCount) {};

		// Iterate through layers, propagating inputs forward for a range of neurons on each layer.
		for (size_t i=0; i<(size_t)nLayers; ++i) {
			networkComponentCount += noThreadsWorking;
		
			layers[i]->calcOutputsVec(metaData.neuronIndexVecVec[i]);
			if (i < (size_t)(nLayers-1)) {
				for (auto j : metaData.neuronIndexVecVec[i]) {
					double inputOutput = layers[i]->getOutput((int)j);
					layers[i+1]->propInputs((int)j, inputOutput);
				}
			}

			// Increment network component finished counter and busy loop until all threads
			// have finished with current layer.
			networkComponentFinishedCount++;
			while (networkComponentFinishedCount.load() < networkComponentCount) {};
		}

		// Have main worker thread calculate filter output and set error.
		networkComponentCount += noThreadsWorking;
		if (metaData.numLayers == (size_t)nLayers) {
			double remover = getOutput(0);
			f_nn = delayed_signal - remover;
			setError(f_nn);
		}
		networkComponentFinishedCount++;
		while (networkComponentFinishedCount.load() < networkComponentCount) {};

		// Propagate error backwards through layers.
		double tempError = 0;
		double tempWeight = 0;
		for (size_t i = (size_t)(nLayers-1); i > 0 ; --i){
			networkComponentCount += noThreadsWorking;
			
			for (auto k : metaData.neuronIndexVecVec[i-1]){
				double sum = 0.0;
				for (int j = 0; j < layers[i]->getnNeurons(); j++){
					tempError = layers[i]->getNeuron(j)->getError();
					tempWeight = layers[i]->getWeights(j, (int)k);
					sum += (tempError * tempWeight);
				}
				assert(std::isfinite(sum));
				layers[i-1]->getNeuron((int)k)->setBackpropError(sum);
			}

			// Increment network component finished counter and busy loop until all threads
			// have finished with current layer.
			networkComponentFinishedCount++;
			while (networkComponentFinishedCount.load() < networkComponentCount) {};
		}

		// Update weights. No need for thread synchronisation inside the for loop here.
		networkComponentCount += noThreadsWorking;
		for (int i = nLayers-1; i >= 0; --i) {
			for (auto j : metaData.neuronIndexVecVec[i]) {
				layers[i]->getNeuron((int)j)->updateWeights();
			}
		}
		networkComponentFinishedCount++;
		while (networkComponentFinishedCount.load() < networkComponentCount) {};

		// If this is the main worker thread, which processes the single neuron on the last layer,
		// finish by notifying the main thread that we are finished filtering.
		if (metaData.numLayers == (size_t)nLayers) {
			{
				std::lock_guard<std::mutex> lk(filterFinishedMtx);
				filterFinishedCond = true;
			}
			filterFinishedCV.notify_one();
		}		
	}
}

//*************************************************************************************
//back propagation of error
//*************************************************************************************

void Net::setError(double _leadError){
	/* this is only for the final layer */
	theLeadError = _leadError;
	//cout<< "lead Error: " << theLeadError << endl;
	layers[nLayers-1]->setError(theLeadError);
}

void Net::propErrorBackward(){
	double tempError = 0;
	double tempWeight = 0;
	for (int i = nLayers-1; i > 0 ; i--){
		for (int k = 0; k < layers[i-1]->getnNeurons(); k++){
			double sum = 0.0;
			for (int j = 0; j < layers[i]->getnNeurons(); j++){
				tempError = layers[i]->getNeuron(j)->getError();
				tempWeight = layers[i]->getWeights(j,k);
				sum += (tempError * tempWeight);
			}
			assert(std::isfinite(sum));
			layers[i-1]->getNeuron(k)->setBackpropError(sum);
		}
	}
}

void Net::propModulatedHebb(float modulator){
	double tempWeight = 0;
	for (int i = nLayers-1; i > 0 ; i--){
		for (int k = 0; k < layers[i-1]->getnNeurons(); k++){
			double sum = 0.0;
			for (int j = 0; j < layers[i]->getnNeurons(); j++){
				tempWeight = layers[i]->getWeights(j,k);
				sum += (modulator * tempWeight);
			}
			assert(std::isfinite(sum));
			layers[i-1]->getNeuron(k)->setBackpropError(sum);
		}
	}
}

//*************************************************************************************
//exploding/vanishing gradient:
//*************************************************************************************

double Net::getGradient(Layer::whichGradient _whichGradient) {
	for (int i=0; i<nLayers; i++) {
		errorGradient[i] = layers[i]->getGradient(_whichGradient);
	}
	double gradientRatio = errorGradient[nLayers -1] / errorGradient[0] ; ///errorGradient[0];
	assert(std::isfinite(gradientRatio));
	return gradientRatio;
}

//*************************************************************************************
//learning:
//*************************************************************************************

void Net::updateWeights(){
	for (int i=nLayers-1; i>=0; i--){
		layers[i]->updateWeights();
	}
}

//*************************************************************************************
// getters:
//*************************************************************************************

double Net::getOutput(int _neuronIndex){
	return (layers[nLayers-1]->getOutput(_neuronIndex));
}

double Net::getSumOutput(int _neuronIndex){
	return (layers[nLayers-1]->getSumOutput(_neuronIndex));
}

int Net::getnLayers(){
	return (nLayers);
}

int Net::getnInputs(){
	return (nInputs);
}

Layer* Net::getLayer(int _layerIndex){
	assert(_layerIndex<nLayers);
	return (layers[_layerIndex]);
}

double Net::getWeightDistance(){
	double weightChange = 0 ;
	double weightDistance =0;
	for (int i=0; i<nLayers; i++){
		weightChange += layers[i]->getWeightChange();
	}
	weightDistance=sqrt(weightChange);
	// cout<< "Net: WeightDistance is: " << weightDistance << endl;
	return (weightDistance);
}

double Net::getLayerWeightDistance(int _layerIndex){
	return layers[_layerIndex]->getWeightDistance();
}

double Net::getWeights(int _layerIndex, int _neuronIndex, int _weightIndex){
	double weight=layers[_layerIndex]->getWeights(_neuronIndex, _weightIndex);
	return (weight);
}

int Net::getnNeurons(){
	return (nNeurons);
}

//*************************************************************************************
//saving and inspecting
//*************************************************************************************

void Net::saveWeights(){
	for (int i=0; i<nLayers; i++){
		layers[i]->saveWeights();
	}
}


void Net::snapWeights(string prefix, string _trial, int _subject){
	for (int i=0; i<nLayers; i++){
		layers[i]->snapWeights(prefix, _trial, _subject);
	}
}

void Net::snapWeightsMatrixFormat(string prefix){
	layers[0]->snapWeightsMatrixFormat(prefix);
}

void Net::printNetwork(){
	cout<< "This network has " << nLayers << " layers" <<endl;
	for (int i=0; i<nLayers; i++){
		cout<< "Layer number " << i << ":" <<endl;
		layers[i]->printLayer();
	}
	cout<< "The output(s) of the network is(are):";
	for (int i=0; i<nOutputs; i++){
		cout<< " " << this->getOutput(i);
	}
	cout<<endl;
}
