#include "Net.h"
#include "netStructure/FFLayer/FFLayer.h"
#include <iostream>
#include <cstdlib>

Net::Net(std::vector<std::pair<int, int>> structure, int seed){
    for(int i = 0; i < structure.size(); i ++){
        if(structure[i].first == 0){
            FFLayer * layer = new FFLayer(i, this, structure[i].second, nullptr);
            this->layers.push_back(layer);
            if(layer->idInNet != 0){
                layer->initConnections(seed);
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
