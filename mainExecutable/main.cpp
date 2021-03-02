#include <iostream>
#include <vector>
#include "geneticLearning/GeneticLearningUnit.h"

int main(){
       
    
    std::vector< std::vector < int > > structure = {{1, 2, 3, 3, 1}, {1, 1, 3, 3, 2}, {0, 4, 25}, {0, 4, 4}, {0, 4, 4}, {0, 4, 4}};
    GeneticLearningUnit learningUnit = GeneticLearningUnit(structure, 1, 1);


   // n.run(input);
   // Tensor t = Tensor(9, 9, 1);
   // n.run(t);
   // std::vector<float> result = n.getResult();
   // for(auto r : result){
   //     std::cout<<r<<" ";
    //}
    std::cout<<"dupa\n";
    
}