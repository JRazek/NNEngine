//
// Created by jrazek on 09.08.2021.
//

#ifndef NEURALNETLIBRARY_QUADTREE_H
#define NEURALNETLIBRARY_QUADTREE_H
#include <unordered_set>
#include <unordered_map>

class QuadTree {
public:
    const float posX, posY;
    const float sizeX, sizeY;
    int pointCount;
    const int pointsLimit;

    QuadTree(float sizeX, float sizeY);
    void insertPoint(float x, float y);
    void removePoint(float x, float y);

    std::unordered_map<float, std::unordered_set<float>> points;

    ~QuadTree();
    [[nodiscard]] bool belongs(float x, float y) const;
private:

    QuadTree * parent;

    QuadTree * NW = nullptr;
    QuadTree * NE = nullptr;
    QuadTree * SW = nullptr;
    QuadTree * SE = nullptr;

    void getChildrenPoints(std::unordered_map<float, std::unordered_set<float>> &pointsSet);
    std::pair<int, int> getNearestNeighbour(const std::pair<int, int> &point);
    QuadTree(float posX, float posY, float sizeX, float sizeY, int pointsLimit, QuadTree * parent);
};


#endif //NEURALNETLIBRARY_QUADTREE_H
