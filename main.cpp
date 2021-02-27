#include <iostream>
#include "neuralNetwork/Net.h"

int main(){
    Net n = Net({{0, 4, 4}, {0, 4, 4}, {0, 4, 4}});

    std::vector<float> input = {1, 2, 3, 4};
    n.run(input);
    std::cout<<"";
}