/**
 * @file    directional_audio.cpp
 * @author  Adam Englebright
 * @date    04.05.2024
 * @brief   Program using DNF in realtime to create directional audio.
 */

#include "alsa_cpp_wrapper.hpp"
#include <cstdint>
#include <iostream>
#include "../dnf.h"
#include "Fir1.h"

int main() {
  // Parameter variables
  unsigned int rate_requested = 32000;
  snd_pcm_uframes_t period_size_requested = 128;
  std::string capture_device = "hw:0,0";
  std::string playback_device = "hw:0,0";

  int taps = 160;
  double learning_rate = 0.05;
  int dnf_layers = 1;
  Neuron::actMethod dnf_act_method = Neuron::Act_Tanh;
  bool dnf_debug_output = false;
  unsigned char dnf_threads = 1;

  bool filter = true;
  bool filter_with_dnf = true;
  
  /* Create our object, with appropriate ALSA PCM identifiers,
   * sample rate, buffer data organization, and period size in frames.
   * The ALSA PCM identifiers will be "hw:0,0" on the pi, but will
   * likely be different on your own hardware. Use "aplay -l" and
   * "arecord -l" on the terminal to find the appropriate id
   * for playback and capture. The first number in the id is the
   * card number, and the second is the device number. */
  Alsa audio(playback_device, capture_device, rate_requested, SND_PCM_ACCESS_RW_INTERLEAVED, period_size_requested);

  // We can get various bits of data from the "audio" object:
  unsigned int rate_actual = audio.getRate();
  snd_pcm_uframes_t period_size_actual = audio.getPeriodSize();

  std::cout << "Period size = " << period_size_actual << std::endl;

  // Most importantlly, we can get a pointer to the buffer storing each period of audio captured
  char *buffer = audio.getBufPtr();
  size_t buffer_size = audio.getBufSize(); // Need to know the size of the buffer in bytes

  std::cout << "Buffer size = " << buffer_size << std::endl;

  // Create our DNF with appropriate settings:
  DNF dnf(dnf_layers, taps, rate_actual, dnf_act_method, dnf_debug_output, dnf_threads);
  dnf.getNet().setLearningRate(learning_rate, 0);

  // Create LMS filter with corresponding primary signal delay line:
  Fir1 lms(taps);
  lms.setLearningRate(learning_rate);
  boost::circular_buffer<double> primary_delay((taps/2 >= 1 ? taps/2 : 1), 0);
  
  //audio.start(); // Start the capture and playback devices (now using automatic start as it seems to handle xruns better)
  while (true) {
    audio.capturePeriod(); // Capture a period

    if (filter) {
      // Each frame has 2 samples (L+R), each 2 bytes in size (S16_LE).
      // Therefore, the for loop must increment 4 bytes at a time.
      // If indexing the buffer, cast to a int16_t*, then this means increments of 2.
      for (int i = 0; i < buffer_size/2; i += 2) {
	// Convert samples to doubles
	double left_sample = (double)( ( (int16_t*)buffer )[i] ) / 32768.0;
	double right_sample = (double)( ( (int16_t*)buffer )[i+1] ) / 32768.0;

	// Filter pre-processing
	double primary = (left_sample + right_sample) / 2;
	double reference = (left_sample - right_sample) / 2;

	// Filter using DNF or LMS
	double output;
	if (filter_with_dnf)
	{
	  output = dnf.filter(primary, reference);
	}
	else
	{
	  primary_delay.push_back(primary);
	  double delayed_primary = primary_delay[0];
	  output = delayed_primary - lms.filter(reference);
	  lms.lms_update(output);
	}

	// Convert filtered signal back to S16_LE and put back in buffer for playback
	int16_t formatted_output = output * 32768.0;
	((int16_t*)buffer)[i] = formatted_output;
	((int16_t*)buffer)[i+1] = formatted_output;
      }
    }
    
    audio.playbackPeriod(); // Playback the captured period
  }
}
