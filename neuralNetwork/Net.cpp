#include <Net.h>
#include <netStructure/FFLayer/FFLayer.h>
#include <activations/SigmoidFunction.h>
#include <activations/ReLUFunction.h>
#include <iostream>
#include <cstdlib>
#include <stdexcept>

Net::Net(std::vector<std::vector<int>> structure, int seed){
    for(int i = 0; i < structure.size(); i ++){
        if(structure[i].size() <= 0){
            throw std::invalid_argument( "wrong description in layer " + i );
            return;
        }
        if(structure[i][settingsLayerTypeInd] == 0){
            if(structure[i].size() <= 2){
                throw std::invalid_argument( "wrong description in layer " + i );
                return;
            }
            FFLayer * layer = new FFLayer(i, this, structure[i][settingsInputSizeInd], structure[i][settingsLayerSizeInd], new SigmoidFunction());
            this->layers.push_back(layer);
            layer->initConnections(seed);
        }
        if(structure[i][settingsLayerTypeInd] == 1){
            //covolution type
        }
    }
}
Net::~Net(){
    for(auto l : layers){
        delete l;
    }
}
void Net::run(std::vector<float> input){
    for(auto l : this->layers){
       FFLayer * ff = (FFLayer *) l;
       ff->run(input);
       input = ff->outputVector;
    }
}