#include "Layer.h"
#include "Neuron.h"

#include <cstddef>
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
#include <fstream>

//*************************************************************************************
// constructor de-constructor
//*************************************************************************************

Layer::Layer(int _nNeurons, int _nInputs,
             boost::circular_buffer<double> &_inputs, int _subject,
             string _trial)
  : inputs(_inputs),
    outputs(_nNeurons, 0)
{
    subject = _subject;
    trial = _trial;
    nNeurons = _nNeurons; // number of neurons in this layer
    nInputs = _nInputs; // number of inputs to each neuron
    neurons = new Neuron*[(unsigned)nNeurons];
    /* dynamic allocation of memory to n number of
     * neuron-pointers and returning a pointer, "neurons",
     * to the first element */
    for (int i=0;i<nNeurons;i++){
	    neurons[i]=new Neuron(nInputs, inputs, outputs);
    }
    /* each element of "neurons" pointer is itself a pointer
     * to a neuron object with specific no. of inputs*/
     //cout << "layer" << endl;
}

Layer::~Layer(){
    for(int i=0;i<nNeurons;i++) {
        delete neurons[i];
    }
    delete[] neurons;
    /* it is important to delete any dynamic
     * memory allocation created by "new" */
}

//*************************************************************************************
//initialisation:
//*************************************************************************************

void Layer::initLayer(int _layerIndex, Neuron::weightInitMethod _wim, Neuron::biasInitMethod _bim, Neuron::actMethod _am){
    myLayerIndex = _layerIndex;
    for (int i=0; i<nNeurons; i++){
        neurons[i]->initNeuron(i, myLayerIndex, _wim, _bim, _am);
    }
}

void Layer::setlearningRate(double _w_learningRate, double _b_learningRate){
    for (int i=0; i<nNeurons; i++){
        neurons[i]->setLearningRate(_w_learningRate,_b_learningRate);
    }
}

//*************************************************************************************
//forward propagation of inputs:
//*************************************************************************************

void Layer::setInputs(const boost::circular_buffer<double>& _inputs, const double scale, const int n) {
	/*this is only for the first layer*/
	for (int j=0; j< (n < 0 ? nInputs:n); j++){
		inputs[j] = _inputs[j] * scale; //take this input value
	}
}

void Layer::setInputsMT(const boost::circular_buffer<double>& _inputs, size_t threadID, size_t nThreads, const double scale, const int n) {
	/*this is only for the first layer*/
	for (int j=(int)threadID; j< (n < 0 ? nInputs:n); j+=(int)nThreads){
		inputs[j] = _inputs[j] * scale; //take this input value
	}
}

void Layer::propInputs(int _index, double _value){
	assert((_index>=0)&&(_index<nInputs));
	inputs[_index] = _value;
}

void Layer::calcOutputs(){
	for (int i=0; i<nNeurons; i++){
		layerHasReported = neurons[i]->calcOutput(layerHasReported);
	}
}

void Layer::calcOutputsMT(const size_t _startIndex, const size_t _endIndex) {
	//std::cout << "In calcOutputsMT. startIndex = " << _startIndex << ", endIndex = " << _endIndex << std::endl;
	for (size_t i=_startIndex; i<=_endIndex; i++) {
		layerHasReported = neurons[i]->calcOutput(layerHasReported);
	}
}

void Layer::calcOutputsVec(const std::vector<size_t>& neuronIndexVec) {
	for (auto index : neuronIndexVec) {
		layerHasReported = neurons[index]->calcOutput(layerHasReported);
	}
}

//*************************************************************************************
//back propagation of error:
//*************************************************************************************

void Layer::setError(double _backwardError){
    /* this is only for the final layer */
    backwardError = _backwardError;
    for (int i=0; i<nNeurons; i++){
        neurons[i]->setError(backwardError);
    }
}

//*************************************************************************************
//exploding/vanishing gradient:
//*************************************************************************************

double Layer::getGradient(whichGradient _whichGradient) {
    averageError = 0;
    maxError = -100;
    minError = 100;
    switch(_whichGradient){
        case exploding:
            for (int i=0; i<nNeurons; i++){
                maxError = max(maxError, neurons[i]->getError());
            }
            return maxError;
            break;
        case average:
            for (int i=0; i<nNeurons; i++){
                averageError += neurons[i]->getError();
            }
            return averageError/nNeurons;
            break;
        case vanishing:
            for (int i=0; i<nNeurons; i++){
                minError = min(minError, neurons[i]->getError());
            }
            return minError;
            break;
    }
    return 0;
}

//*************************************************************************************
//learning:
//*************************************************************************************

void Layer::updateWeights(){
    for (int i=0; i<nNeurons; i++){
        neurons[i]->updateWeights();
    }
}

//*************************************************************************************
//getters:
//*************************************************************************************

Neuron* Layer::getNeuron(int _neuronIndex){
    assert(_neuronIndex < nNeurons);
    return (neurons[_neuronIndex]);
}

double Layer::getSumOutput(int _neuronIndex){
    return (neurons[_neuronIndex]->getSumOutput());
}

double Layer::getWeights(int _neuronIndex, int _weightIndex){
    return (neurons[_neuronIndex]->getWeights(_weightIndex));
}

double Layer::getInitWeight(int _neuronIndex, int _weightIndex){
    return (neurons[_neuronIndex]->getInitWeights(_weightIndex));
}

double Layer::getWeightChange(){
    weightChange=0;
    for (int i=0; i<nNeurons; i++){
        weightChange += neurons[i]->getWeightChange();
    }
    //cout<< "Layer: WeightChange is: " << weightChange << endl;
    return (weightChange);
}

double Layer::getWeightDistance(){
    return sqrt(weightChange);
}

double Layer::getOutput(int _neuronIndex){
    return outputs[_neuronIndex];
}

boost::circular_buffer<double>& Layer::getOutputArray(){
    return outputs;
}

int Layer::getnNeurons(){
    return (nNeurons);
}

//*************************************************************************************
//saving and inspecting
//*************************************************************************************

void Layer::saveWeights(){
    for (int i=0; i<nNeurons; i++){
        neurons[i]->saveWeights();
    }
}

void Layer::snapWeights(string prefix, string _trial, int _subject){
    subject = _subject;
    trial = _trial;
    std::fstream wfile;
    string name = prefix+"/subject" + std::to_string(subject) + "/grayLayer" + std::to_string(myLayerIndex+1) + "_" + trial + "_weights.csv";
    wfile.open(name, fstream::out);
    if (!wfile || !wfile) {
        cout << "Unable to open grayScale files";
        exit(1); // terminate with error
    }
    for (int i=0; i<nNeurons; i++){
        for (int j=0; j<nInputs; j++){
            wfile << neurons[i]->getWeights(j) << " ";
        }
        wfile << "\n";
    }
    wfile.close();
}

void Layer::snapWeightsMatrixFormat(string prefix){
    std::ofstream wfile;
    string name = prefix+"/subject" + std::to_string(subject) + "/MATRIX_Layer"
                  + "_Subject" + std::to_string(subject)
                  + "_" + trial;
    name += ".csv";
    wfile.open(name);
    wfile << "[" << nNeurons << "," << nInputs << "]";
    wfile << "(";
    for (int i=0; i<nNeurons; i++){
        if (i == 0){
            wfile << "(";
        }else{
            wfile << ",(";
        }
        for (int j=0; j<nInputs; j++){
            if (j == 0){
                wfile << neurons[i]->getWeights(j);
            }else{
                wfile << "," << neurons[i]->getWeights(j);
            }
        }
        wfile << ")";
        //wfile << "\n";
    }
    wfile << ")";
    wfile.close();
}

void Layer::printLayer(){
    cout<< "\t This layer has " << nNeurons << " Neurons" <<endl;
    cout<< "\t There are " << nInputs << " inputs to this layer" <<endl;
    for (int i=0; i<nNeurons; i++){
        cout<< "\t Neuron number " << i << ":" <<endl;
        neurons[i]->printNeuron();
    }

}
