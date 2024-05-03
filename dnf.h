#ifndef _DNF_H
#define _DNF_H

#include <boost/circular_buffer.hpp>

#include "dnf/Neuron.h"
#include "dnf/Layer.h"
#include "dnf/Net.h"

/**
 * Main Deep Neuronal Network main class.
 * It's designed to be as simple as possible with
 * only a few parameters as possible.
 **/
class DNF {
public:
	/**
	 * Constructor which sets up the delay lines, network layers
	 * and also calculates the number of neurons per layer so
	 * that the final layer always just has one neuron.
	 * \param NLAYERS Number of layers
	 * \param numTaps Number of taps for the delay line feeding into the 1st layer
	 * \param fs Sampling rate of the signals used in Hz.
	 * \param am The activation function for the neurons. Default is tanh.
	 * \param _nThreads Number of threads for parallel processing. Default is 1 thread for single threaded operation. A value of 0 will be converted to 1.
	 **/
	DNF(const int NLAYERS,
	    const int numTaps,
	    const double fs,
	    const Neuron::actMethod am = Neuron::Act_Tanh,
	    const bool debugOutput = false,
	    const unsigned char _nThreads = 1
		) : noiseDelayLineLength(numTaps),
		    signalDelayLineLength(noiseDelayLineLength / 2),
		    signal_delayLine(signalDelayLineLength, 0),
		    nNeurons(new int[NLAYERS]),
		    noise_delayLine(noiseDelayLineLength, 0),
		    nThreads(_nThreads) {
		// Check number of threads is >0
		if (nThreads < 1)
			nThreads = 1;
		
		// calc an exp reduction of the numbers always reaching 1
		double b = exp(log(noiseDelayLineLength)/(NLAYERS-1));
		for(int i=0;i<NLAYERS;i++) {
			nNeurons[i] = ceil(noiseDelayLineLength / pow(b,i));
			if (i == (NLAYERS-1)) nNeurons[i] = 1;
			//std::cout << "Neurons on layer " << i << ": " << nNeurons[i] << std::endl;
		}
		
		//create the neural network
		NNO = new Net(NLAYERS, nNeurons, noiseDelayLineLength, noise_delayLine, 0, "", nThreads);
		
		//setting up the neural networks
		for(int i=0;i<NLAYERS;i++) {
			NNO->getLayer(i)->initLayer(i,Neuron::W_RANDOM, Neuron::B_NONE, am);
			if (debugOutput) {
				fprintf(stderr,"Layer %d has %d neurons. act = %d\n",i,nNeurons[i],am);
			}
		}
	}

	enum ErrorPropagation { Backprop = 0, ModulatedHebb = 1 };

	void setErrorPropagation(const ErrorPropagation e) {
		errorPropagation = e;
	}

	/**
	 * Realtime sample by sample filtering operation
	 * \param signal The signal contaminated with noise. Should be less than one.
	 * \param noise The reference noise. Should be less than one.
	 * \param doBackProp Flag to indicate if error backprop and weight updating should be done for this filter operation.
	 * \returns The filtered signal where the noise has been removed by the DNF.
	 **/
	double filter(const double signal, const double noise, const bool doBackProp = true) {
		signal_delayLine.push_back(signal);
		const double delayed_signal = signal_delayLine[0];
		
		noise_delayLine.push_front(noise / (double)noiseDelayLineLength);

		if (nThreads == 1) {
			// NOISE INPUT TO NETWORK
			//std::cout << "Setting inputs\n";
			//NNO->setInputs(noise_delayLine);
			
			//std::cout << "Propagating inputs forward\n";
			NNO->propInputs();
			
			// REMOVER OUTPUT FROM NETWORK
			remover = NNO->getOutput(0);
			f_nn = delayed_signal - remover;

			if (doBackProp) {
				// FEEDBACK TO THE NETWORK 
				NNO->setError(f_nn);
				//std::cout << "Propagating error backwards\n";
				switch (errorPropagation) {
				case Backprop:
				default:
					NNO->propErrorBackward();
					break;
				case ModulatedHebb:
					NNO->propModulatedHebb(f_nn);
					break;
				}
				//std::cout << "Updating weights\n";
				NNO->updateWeights();
			}
			
			return f_nn;
		} else {
			return NNO->filterMT(delayed_signal, doBackProp);
		}
	}

	/**
	 * Returns a reference to the whole neural network.
	 * \returns A reference to Net.
	 **/
	inline Net& getNet() const {
		return *NNO;
	}

	/**
	 * Returns the length of the delay line which
	 * delays the signal polluted with noise.
	 * \returns Number of delay steps in samples.
	 **/
	inline const int getSignalDelaySteps() const {
		return signalDelayLineLength;
	}

	/**
	 * Returns the delayed with noise polluted signal by the delay 
	 * indicated by getSignalDelaySteps().
	 * \returns The delayed noise polluted signal sample.
	 **/
	inline const double getDelayedSignal() const {
		return signal_delayLine[0];
	}

	/**
	 * Returns the remover signal.
	 * \returns The current remover signal sample.
	 **/
	inline const double getRemover() const {
		return remover;
	}

	/**
	 * Returns the output of the DNF: the the noise
	 * free signal.
	 * \returns The current output of the DNF which is idential to filter().
	 **/
	inline const double getOutput() const {
		return f_nn;
	}

	/**
	 * Frees the memory used by the DNF.
	 **/
	~DNF() {
		delete NNO;
		delete[] nNeurons;
	}

private:
	Net *NNO;
	int noiseDelayLineLength;
	int signalDelayLineLength;
	boost::circular_buffer<double> signal_delayLine;
	boost::circular_buffer<double> noise_delayLine;
	int* nNeurons;
	double remover = 0;
	double f_nn = 0;
	ErrorPropagation errorPropagation = Backprop;
	unsigned char nThreads;
};

#endif
