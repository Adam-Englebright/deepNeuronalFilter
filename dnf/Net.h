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
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include "Layer.h"

/** Net is the main class used to set up a neural network used for
 * closed-loop Deep Learning. It initialises all the layers and the
 * neurons internally.
 *
 * (C) 2019,2020, Bernd Porr <bernd@glasgowneuro.tech>
 * (C) 2019,2020, Sama Daryanavard <2089166d@student.gla.ac.uk>
 *
 * GNU GENERAL PUBLIC LICENSE
 **/
class Net {

// Need this initial private to declare this struct before it's used
// in the declaration of propInputsThread.
private:
	/**
	 * Struct containing meta data for each forward propagation worker thread.
	 */
	struct ThreadMetaData {
		/**
		 * Vector of vectors of neuron indices to process for each layer.
		 */
		std::vector<std::vector<size_t>> neuronIndexVecVec;

		/**
		 * Vector containing the number of threads working on each layer.
		 */
		std::vector<size_t> numThreadsVec;

		/**
		 * Number of layers this thread will work on.
		 * I.e. neuronIndexVecVec.size()
		 */
		size_t numLayers = 0;
	};

public:

/** Constructor: The neural network that performs the learning.
 * \param _nLayers Total number of hidden layers, excluding the input layer
 * \param _nNeurons A pointer to an int array with number of
 * neurons for all layers need to have the length of _nLayers.
 * \param _nInputs Number of Inputs to the network
 * \param _inputBuffer Reference to a circular buffer of inputs.
 * \param _nThreads Number of threads for processing forward propagation.
 **/
	Net(const int _nLayers,
	    const int * const _nNeurons,
	    const int _nInputs,
	    boost::circular_buffer<double>& _inputBuffer,
	    const int _subject,
	    const string _trial,
	    const unsigned char _nThreads = 1);

/**
 * Destructor
 * De-allocated any memory
 **/
	~Net();

/** Dictates the initialisation of the weights and biases
 * and determines the activation function of the neurons.
 * \param _wim weights initialisation method,
 * see Neuron::weightInitMethod for different options
 * \param _bim biases initialisation method,
 * see Neuron::biasInitMethod for different options
 * \param _am activation method,
 * see Neuron::actMethod for different options
 **/
	void initNetwork(Neuron::weightInitMethod _wim, Neuron::biasInitMethod _bim, Neuron::actMethod _am);
	
/** Sets the learning rate.
 * \param _learningRate Sets the learning rate for
 * all layers and neurons.
 **/
	void setLearningRate(double _w_learningRate, double _b_learningRate);
	
/** Sets the inputs to the network in each iteration
 * of learning, needs to be placed in an infinite loop.
 * @param _inputs A pointer to the array of inputs
 */
	void setInputs(const boost::circular_buffer<double>& _inputs, const double scale = 1.0, const unsigned int offset = 0, const int n = -1);
/**
 * It propagates the inputs forward through the network.
 */
	void propInputs();

	/**
	 * Filters a sample, multi-threaded.
	 * @param _delated_signal Delayed input signal.
	 * @return Result of the filter operation.
	 */
	double filterMT(double _delayed_signal);

	/**
	 * Filter worker thread.
	 * @param mataData Data required by each thread, namely the neurons of each layer that they are to work on.
	 */
	void filterThread(ThreadMetaData metaData);

	/**
	 * Propagates the error backward
	 **/
	void propErrorBackward();


	/**
	 * Propagates the error backward
	 **/
	void propModulatedHebb(float modulator);


/**
 * Sets the error at the output layer to be propagated backward.
 * @param _leadError The closed-loop error for learning
 */
	void setError(double _leadError);

/**
 * It provides a measure of how the magnitude of the error changes through the layers
 * to alarm for vanishing or exploding gradients.
 * \param _whichError choose what error to monitor, for more information see Neuron::whichError
 * \param _whichGradient choose what gradient of the chosen error to monitor,
 * for more information see Layer::whichGradient
 * @return Returns the ratio of the chosen gradient in the last layer to the the first layer
 */
	double getGradient(Layer::whichGradient _whichGradient);

/**
 * Requests that all layers perform one iteration of learning
 */
	void updateWeights();

/**
 * Allows Net to access each layer
 * @param _layerIndex the index of the chosen layer
 * @return A pointer to the chosen Layer
 */
	Layer *getLayer(int _layerIndex);
/**
 * Allows the user to access the activation output of a specific neuron in the output layer only
 * @param _neuronIndex The index of the chosen neuron
 * @return The value at the output of the chosen neuron
 */
	double getOutput(int _neuronIndex);
/**
 * Allows the user to access the weighted sum output of a specific neuron in output layer only
 * @param _neuronIndex The index of the chosen neuron
 * @return The value at the sum output of the chosen neuron
 */
	double getSumOutput(int _neuronIndex);

/**
 * Informs on the total number of hidden layers (excluding the input layer)
 * @return Total number of hidden layers in the network
 */
	int getnLayers();
/**
 * Informs on the total number of inputs to the network
 * @return Total number of inputs
 */
	int getnInputs();

/**
 * Allows for monitoring the overall weight change of the network.
 * @return returns the Euclidean wight distance of all neurons in the network from their initial value
 */
	double getWeightDistance();

/**
 * Allows for monitoring the weight change in a specific layer of the network.
 * @param _layerIndex The index of the chosen layer
 * @return returns the Euclidean wight distance of neurons in the chosen layer from their initial value
 */
	double getLayerWeightDistance(int _layerIndex);
/**
 * Grants access to a specific weight in the network
 * @param _layerIndex Index of the layer that contains the chosen weight
 * @param _neuronIndex Index of the neuron in the chosen layer that contains the chosen weight
 * @param _weightIndex Index of the input to which the chosen weight is assigned
 * @return returns the value of the chosen weight
 */
	double getWeights(int _layerIndex, int _neuronIndex, int _weightIndex);

/**
 * Informs on the total number of neurons in the network
 * @return The total number of neurons
 */
	int getnNeurons();

/**
 * Saves the temporal changes of all weights in all neurons into files
 */
	void saveWeights();
/**
 * Snaps the final distribution of all weights in a specific layer,
 * this is overwritten every time the function is called
 */
	void snapWeights(string prefix, string _trial, int _subject);
	void snapWeightsMatrixFormat(string prefix);
/**
 * Prints on the console a full tree of the network with the values of all weights and outputs for all neurons
 */
	void printNetwork();

private:

	/**
	 * Reference to the input circular buffer for the network.
	 */
	boost::circular_buffer<double>& inputBuffer;

	/**
	 * Total number of hidden layers
	 */
	int nLayers = 0;

        /**
         * The error
         **/
        double theLeadError = 0;

        /**
	 * total number of neurons
	 */
	int nNeurons = 0;

        /**
	 * total number of weights
	 */
	int nWeights = 0;
  
	/**
	 * total number of inputs
	 */
	int nInputs = 0;
  
	/**
	 * total number of outputs
	 */
	int nOutputs = 0;
  
	/**
	 * A double pointer to the layers in the network
	 */
	Layer **layers = 0;
  
	/**
	 * A pointer to the inputs of the network
	 */
	const double *inputs = 0;
  
	/**
	 * A pointer to the gradient of the error
	 */
	double *errorGradient = NULL;

	/**
	 * Mutex for the filterStartCV condition variable used to control the filter threads.
	 */
	std::mutex filterStartMtx;

	/**
	 * Condition variable for controlling the execution of the filter threads.
	 */
	std::condition_variable filterStartCV;

	/**
	 * Flag storing the condition of the filter start.
	 * Set to false by default, when the DNF is idle and no samples have been provided.
	 * Set to true once a sample is provided and propagation through the network
	 * is desired, using the worker threads for this task.
	 */
	bool filterStartCond = false;

	/**
	 * Mutex for the filterFinishedCV condition variable used to signal that the filter
	 * worker threads have finished.
	 */
	std::mutex filterFinishedMtx;

	/**
	 * Condition variable used to signal that the filter worker threads have finished.
	 */
	std::condition_variable filterFinishedCV;

	/**
	 * Flag storing the condition for the progress of the filter worker threads.
	 * False if the worker threads have not finished, true if they have.
	 */
	bool filterFinishedCond = false;

        /**
	 * Counter to track how many filter worker threads have finished with
	 * the network component currently being processed. Used for synchronisation
	 * between threads, to ensure some threads don't run ahead of the others.
	 */
	atomic_size_t networkComponentFinishedCount{0};

	/**
	 * Counter to track how may filter worker threads are ready to start working,
	 * and are now waiting for the signal to start.
	 */
	atomic_size_t filterThreadReadyCount{0};

	/**
	 * Number of threads actually doing work. Not necessarily the number requested.
	 */
	size_t noThreadsWorking = 0;

	/**
	 * Flag to indicate to the filter threads that they should terminate.
	 */
	bool filterThreadsTerm = false;

	/**
	 * Output of the filter.
	 */
	double f_nn;

	/**
	 * Delayed signal value.
	 */
	double delayed_signal;

	/**
	 * A vector/pool of threads for calculating a filter result by running through the network.
	 */
	std::vector<std::thread> filterThreadPool;
};
