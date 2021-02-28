#include <iostream>
#include "neuralNetwork/Net.h"

int main(){
    Net n = Net({{1, 3, 3, 3, 3}, {0, 4, 4}, {0, 4, 4}});

    std::vector<float> input = {1, 2, 3, 4};
   // n.run(input);
    Tensor t = Tensor(9, 9, 3);
    n.run(t);
    std::vector<float> result = n.getResult();
    for(auto r : result){
        std::cout<<r<<" ";
    }
    std::cout<<"";
    
}