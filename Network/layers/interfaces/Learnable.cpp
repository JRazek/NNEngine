//
// Created by user on 21.08.2021.
//
#include "Learnable.h"

cn::Learnable::Learnable(int id, cn::Network &network, int _neuronsCount, const DifferentiableFunction &differentiableFunction) : Layer(id, network),
                                                                                                                                  neuronsCount(_neuronsCount),
                                                                                                                                  activationFunction(
                                                                                         differentiableFunction) {}