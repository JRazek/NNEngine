#include "Net.h"
#include <iostream>

Net::Net(std::vector<std::pair<bool, int>> structure){
    for(int i = 0; i < structure.size(); i ++){
        if(structure[i].first == 0){
            FFLayer * layer = new FFLayer(i, structure[i].second);
            this->layers.push_back((Layer *) layer);
            for(int i = 0; i < structure[i].second; i ++){
                //FFLayer::Neuron::Neuron * n = new FFLayer::Neuron::Neuron(i);
                //todo
            }
        }
    }
}
Net::~Net(){
    for(auto l : layers){
        delete l;
    }
}
