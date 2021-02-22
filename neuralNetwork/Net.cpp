#include "Net.h"
#include <iostream>

Net::Net(std::vector<std::pair<bool, int>> structure){
    for(int i = 0; i < structure.size(); i ++){
        this->layers.push_back((Layer *) new FFLayer(i, structure[i].second));
    }
}
Net::~Net(){
    for(auto l : layers){
        delete l;
    }
}
