//
// Created by jrazek on 27.07.2021.
//

#include "Layer.h"

cn::Layer::Layer(int _id, cn::Network *network): id(_id), network(network){}

cn::Layer::~Layer() {
    delete output;
}
