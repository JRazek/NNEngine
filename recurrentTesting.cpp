//
// Created by user on 31.07.2021.
//
#include <iostream>
#include "Utils/dataStructures/Tensor.h"
#include "Utils/Utils.h"
#include "Network/Network.h"
#include "Optimizers/MBGD.h"
#include "Optimizers/MomentumGD.h"
#include <opencv2/opencv.hpp>
#include "Utils/Files/CSVReader.h"
#include "Utils/Files/ImageRepresentation.h"
#include "Utils/dataStructures/VectorN.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreorder"

int main(){
    cn::Network network(28, 28, 3, 1);

    const int outputSize = 10;
    network.appendMaxPoolingLayer({2,2});
    network.appendRecurrentLayer();
    network.appendFlatteningLayer();
    network.appendFFLayer(outputSize);
    network.appendSigmoidLayer();
    network.appendRecurrentLayer();
    network.appendFFLayer(outputSize);
    network.appendSigmoidLayer();
    network.appendFFLayer(outputSize);
    network.appendSigmoidLayer();
    network.initRandom();
    network.ready();

    cn::MomentumGD momentumGd(network, 0.7, 0.01);


    return 0;
}
