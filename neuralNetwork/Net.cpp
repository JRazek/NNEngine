#include <Net.h>
#include <netStructure/FFLayer/FFLayer.h>
#include <netStructure/ConvolutionLayer/CLayer.h>
#include <activations/SigmoidFunction.h>
#include <activations/ReLUFunction.h>
#include <iostream>
#include <cstdlib>
#include <stdexcept>

Net::Net(std::vector<std::vector<int>> structure, int seed){
    srand(seed);
    for(int i = 0; i < structure.size(); i ++){
        if(structure[i].size() < 3){
            throw std::invalid_argument( "wrong description in layer " + i );
            return;
        }
        if(structure[i][0] == 0){
            FFLayer * layer = new FFLayer(i, this, structure[i][2], structure[i][1], new SigmoidFunction());
            this->layers.push_back(layer);
            layer->initConnections();
        }
        if(structure[i][0] == 1){
            //todo
            if(structure[i].size() < 5){
                throw std::invalid_argument( "wrong description in layer " + i );
                return;
            }
            CLayer * clayer = new CLayer(i, this, structure[i][1], structure[i][2], structure[i][3], structure[i][4]);
            clayer->initWeights();
            layers.push_back(clayer);
        }
    }
}
Net::~Net(){
    for(auto l : layers){
        delete l;
    }
}

//void run(const CLayer::Tensor &tensorInput);

void Net::run(const Tensor &tensorInput){
    for(int i = 0; i < this->layers.size(); i ++){
        Layer * l = layers[i];
        if (dynamic_cast<CLayer*>(l) != nullptr) {
            CLayer * cl = (CLayer *) l;
            cl->run(tensorInput);
           // input = cl->outputVector;
        }
    }
}