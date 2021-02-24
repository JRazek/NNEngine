#include "Net.h"
#include <iostream>
#include <cstdlib>

Net::Net(std::vector<std::pair<int, int>> structure, int seed){
    for(int i = 0; i < structure.size(); i ++){
        if(structure[i].first == 0){
            FFLayer * layer = new FFLayer(i, structure[i].second);
            this->layers.push_back((Layer *) layer);
            for(int j = 0; j < structure[i].second; j ++){
                FFLayer::Neuron * n = new FFLayer::Neuron(j);
                layer->neurons.push_back(n);
                if(layer->idInNet != 0){
                    srand (seed);
                    float randWeight = (rand() % 100)/100.f;
                    float randBias = (rand() % 1000)/100.f;

                    FFLayer * prevLayer = (FFLayer *)layers[layer->idInNet - 1];
                    for(auto pN : prevLayer->neurons){
                        pN->outputEdges.push_back({n->idInLayer, randWeight});
                    }
                    n->bias = randBias;
                }
                
                std::cout<<"";
            }
        }
        if(structure[i].first == 1){
            //covolution type
        }
    }
}
Net::~Net(){
    for(auto l : layers){
        delete l;
    }
}
