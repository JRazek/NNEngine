//
// Created by jrazek on 09.08.2021.
//

#ifndef NEURALNETLIBRARY_QUADTREE_H
#define NEURALNETLIBRARY_QUADTREE_H


class QuadTree {
public:
    const int posX, posY;
    const int sizeX, sizeY;
    QuadTree(int sizeX, int sizeY);
    void insertPoint(int x, int y);
    void removePoint(int x, int y);

private:

    QuadTree * NW;
    QuadTree * NE;
    QuadTree * SW;
    QuadTree * SE;

    QuadTree(int posX, int posY, int sizeX, int sizeY);
};


#endif //NEURALNETLIBRARY_QUADTREE_H
