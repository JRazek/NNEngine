//
// Created by jrazek on 09.08.2021.
//

#include "QuadTree.h"

QuadTree::QuadTree(int posX, int posY, int sizeX, int sizeY): posX(posX), posY(posY), sizeX(sizeX), sizeY(sizeY) {

}

QuadTree::QuadTree(int sizeX, int sizeY): QuadTree(0, 0, sizeX, sizeY){

}
