#include <iostream>
#include <vector>
#include "geneticLearning/GeneticLearningUnit.h"

int main(){
       
    
    std::vector< std::vector < int > > structure = {{1, 2, 3, 3, 1}, {1, 1, 3, 3, 2}, {2, 2, 2}, {0, 4, 25}, {0, 4, 4}, {0, 4, 4}};
    GeneticLearningUnit learningUnit = GeneticLearningUnit(structure, 1, 1);
   /* Net * n = new Net(structure);
    Tensor t = Tensor(15, 15, 1);

    srand(1);
    for(int z = 0; z < t.getZ(); z++){
        for(int y = 0; y < t.getY(); y ++){
            for(int x = 0; x < t.getX(); x ++){
                t.edit(x, y, z, rand() % 1000);
            }
        }
    }
    n->run(t);
    std::vector<float> result = n->getResult();
    for(auto r : result){
        std::cout<<r<<" ";
    }
    delete n;*/
    learningUnit.start(1);
}