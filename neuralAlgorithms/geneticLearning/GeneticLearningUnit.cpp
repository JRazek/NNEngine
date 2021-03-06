#include "GeneticLearningUnit.h"
#include <stdlib.h>
#include <iostream>
#include <stdexcept>
#include <ncurses.h>
GeneticLearningUnit::GeneticLearningUnit(const std::vector< std::vector < int > > structure, int goalGenerations, int individualsPerGeneration):
    structure(structure), goalGenerations(goalGenerations), individualsPerGeneration(individualsPerGeneration){}
    
void GeneticLearningUnit::start(int seed = 0){
    for(int i = 0; i < this->individualsPerGeneration; i ++){
        srand(seed);
        csnake::SnakeGame * game = new csnake::SnakeGame(this->mapSizeX, this->mapSizeY, this->impenetrableWalls);
        WINDOW * win = nullptr;
        if(i == 0){
            win = initscr();
        }
        csnake::Renderer * renderer = new csnake::Renderer(game, win);
        Net * net = new Net(this->structure, rand() % (int)1e9+7);
        this->currIndividuals.push_back({net, renderer});
    }
    std::vector<bool> deadTab(currIndividuals.size(), false);

    while(1){
        int deadCount = 0;
        for(int i = 0; i < currIndividuals.size(); i ++){
            auto k = currIndividuals[i];
            auto board = k.second->getBoard();
            Tensor input = Tensor(k.second->getGame()->getWidth(), k.second->getGame()->getHeight(), 1);
            if(board.size() != input.getY() || board[0].size() != input.getX()){
                throw std::invalid_argument("board and tensor wont match!");
            }
            for(int y = 0; y < input.getY(); y++){
                for(int x = 0; x < input.getX(); x++){
                    input.edit(x, y, 0, board[y][x]);
                }
            }
            auto gameManager = k.second;
            if(i == 0){
                gameManager->run(true);
            }else{
                gameManager->run(false);
            }
            gameManager->getGame()->update();

            if(!gameManager->getGame()->getSnakeStatus() && deadTab[i] == false){
                deadTab[i] == true;
                deadCount ++;
            }
        }
        if(deadCount == this->currIndividuals.size()){
            break;
        }
    }
}

GeneticLearningUnit::~GeneticLearningUnit(){
    for(auto k : currIndividuals){
        delete k.first;
        delete k.second;
    }
}