#pragma once



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

#include "Neuron.h"

/**
 * This is the class for creating layers that are contained inside the Net class.
 * The Layer instances in turn contain neurons.
 */

class Layer {
public:
	/**
	 * Constructor for Layer: it initialises the neurons internally.
	 * @param _nNeurons Total number of neurons in the layer
	 * @param _nInputs Total number of inputs to that layer
	 * @param _networkInputs Reference to the input circular buffer for the network.
	 */
	Layer(int _nNeurons, int _nInputs, int _subject, string _trial, boost::circular_buffer<double>& _networkInputs);
	/**
	 * Destructor
	 * De-allocated any memory
	 **/
	~Layer();

	/**
	 * Options for what gradient of a chosen error to monitor
	 */
	enum whichGradient {exploding = 0, average = 1, vanishing = 2};

	/**
	 * Initialises each layer with specific methods for weight/bias initialisation and activation function of neurons
	 * @param _layerIndex The index that is assigned to this layer by the Net class
	 * @param _wim weights initialisation method,
	 * see Neuron::weightInitMethod for different options
	 * @param _bim biases initialisation method,
	 * see Neuron::biasInitMethod for different options
	 * @param _am activation method,
	 * see Neuron::actMethod for different options
	 */
	void initLayer(int _layerIndex, Neuron::weightInitMethod _wim, Neuron::biasInitMethod _bim, Neuron::actMethod _am);
	
	/** Sets the learning rate.
	 * @param _learningRate Sets the learning rate for all neurons.
	 **/
	void setlearningRate(double _w_learningRate, double _b_learningRate);
	
	/**
	 * Sets the inputs to all neurons in the first hidden layer only
	 * @param _inputs A reference to a circular buffer of inputs.
	 * @param scale Scale factor applied to inputs. Default 1.
	 * @param n Number of inputs set. Default is the number of inputs to the layer.
	 */
	void setInputs(const boost::circular_buffer<double>& _inputs, const double scale = 1.0, const int n = -1);

	/**
	 * Sets the inputs to a vector of neurons in the first hidden layer only. For multi-threaded use.
	 * @param _inputs A reference to a circular buffer of inputs.
	 * @param threadID Thread index starting from 0.
	 * @param nThreads Number of threads working.
	 * @param scale Scale factor applied to inputs. Default 1.
	 * @param n Number of inputs set. Default is the number of inputs to the layer.
	 */
	void setInputsMT(const boost::circular_buffer<double>& _inputs, size_t threadID, size_t nThreads, const double scale = 1.0, const int n = -1);

	/**
	 * Sets the inputs to all neurons in the deeper layers (excluding the first hidden layer)
	 * @param _index The index of the input
	 * @param _value The value of the input
	 */
	void propInputs(int _index, double _value);
	
	/**
	 * Demands that all neurons in this layer calculate their output
	 */
	void calcOutputs();

	/**
	 * Demands that a range of neurons in this layer calculate their output. For multi-threaded use.
	 * @param _startIndex Index of first neuron to calculate the outputs for.
	 * @param _endIndex Index of the last neuron to calculate the outputs for.
	 */
	void calcOutputsMT(const size_t _startIndex, const size_t _endIndex);

	/**
	 * Demands that a vector of neurons have their outputs calculated. For multi-threaded use.
	 * @param neuronIndexVec Vector of neuron indices to be processed.
	 */
	void calcOutputsVec(const std::vector<size_t>& neuronIndexVec);

	/**
	 * Sets the error to be propagated backward at all neurons in the output layer only.
	 * @param _leadError the error to be propagated backward
	 **/
	void setError(double _error);
	
	/**
	 * Allows for accessing the error that propagates backward in the network
	 * @param _neuronIndex The index from which the error is requested
	 * @return Returns the error of the chosen neuron
	 */
	double getGradient(whichGradient _whichGradient);
	
	/**
	 * Requests that all neurons perform one iteration of learning
	 */
	void updateWeights();

	/**
	 * Allows access to a specific neuron
	 * @param _neuronIndex The index of the neuron to access
	 * @return A pointer to that neuron
	 */
	Neuron *getNeuron(int _neuronIndex);
	
	/**
	 * Reports the number of neurons in this layer
	 * @return The total number of neurons in this layer
	 */
	int getnNeurons();
	
	/**
	 * Allows for accessing the activation of a specific neuron
	 * @param _neuronIndex The index of the neuron
	 * @return the activation of that neuron
	 */
	double getOutput(int _neuronIndex);
	
	/**
	 * Allows for accessing the sum output of any specific neuron
	 * @param _neuronIndex The index of the neuron to access
	 * @return Returns the wighted sum of the inputs to that neuron
	 */
	double getSumOutput(int _neuronIndex);
	
	/**
	 * Allows for accessing any specific weights in the layer
	 * @param _neuronIndex The index of the neuron containing that weight
	 * @param _weightIndex The index of the input to which that weight is assigned
	 * @return Returns the chosen weight
	 */
	double getWeights(int _neuronIndex, int _weightIndex);
	
	/**
	 * Accesses the total sum of weight changes of all the neurons in this layer
	 * @return sum of weight changes all neurons
	 */
	double getWeightChange();
	
	/**
	 * Performs squared root on the weight change
	 * @return The sqr of the weight changes
	 */
	double getWeightDistance();
	
	/**
	 * Reports the global error that is assigned to a specific neuron in this layer
	 * @param _neuronIndex the neuron index
	 * @return the value of the global error
	 */
	double getGlobalError(int _neuronIndex);
	
	/**
	 * Reports the initial value that was assigned to a specific weight at the initialisatin of the network
	 * @param _neuronIndex Index of the neuron containing the weight
	 * @param _weightIndex Index of the weight
	 * @return
	 */
	double getInitWeight(int _neuronIndex, int _weightIndex);
	
	/**
	 * Saves the temporal weight change of all weights in all neurons into files
	 */	
	void saveWeights();
	
	/**
	 * Snaps the final distribution of weights in a specific layer,
	 * this is overwritten every time the function is called
	 */
	void snapWeights(string prefix, string _trial, int _subject);
	
	void snapWeightsMatrixFormat(string prefix);
	
	/**
	 * Prints on the console a full tree of this layer with the values of all weights and outputs for all neurons
	 */
	void printLayer();

private:
	// initialisation:
	int nNeurons = 0;
	int nInputs = 0;
	double learningRate = 0;
	int myLayerIndex = 0;
	Neuron **neurons = 0;
	double *inputs = 0;
	boost::circular_buffer<double>& networkInputs;
    
	int layerHasReported = 0;

	//back propagation of error:
	double backwardError = 0;

	//exploding vasnishing gradient:
	double averageError = 0;
	double maxError = 0;
	double minError = 0;

	int subject = 0;
	string trial = "0";

	//learning:
	double weightChange=0;
};
