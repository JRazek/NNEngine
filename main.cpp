#include <iostream>
#include "neuralNetwork/Net.h"

int main(){
    //{ layerNum:{type, neuronsSize, inputSize} }
    //{ layerNum:{type, tensorsCount, matrixSizeX, matrixSizeY, tensorDepth} }
    //todo fix the read
    int seed;
    std::cin >> seed;
    Net n = Net({{1, 1, 3, 3, 1}, {0, 4, 49}, {0, 4, 4}}, seed);

    std::vector<float> input = {1, 2, 3, 4};
   // n.run(input);
    Tensor t = Tensor(9, 9, 1);
    n.run(t);
    std::vector<float> result = n.getResult();
    for(auto r : result){
        std::cout<<r<<" ";
    }
    std::cout<<"";
    
}