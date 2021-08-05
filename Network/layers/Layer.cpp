//
// Created by jrazek on 27.07.2021.
//

#include "Layer.h"

cn::Layer::Layer(int id, cn::Network *network): id(id), network(network){}

cn::Layer::~Layer() = default;
