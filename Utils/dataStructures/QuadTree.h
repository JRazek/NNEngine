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
    const int level;
    int pointCount;
    const int pointsLimit;
    const int levelLimit;

    QuadTree(float sizeX, float sizeY, int levelLimit);
    void insertPoint(float x, float y);
    void removePoint(float x, float y);

    std::unordered_map<float, std::unordered_set<float>> points;

    ~QuadTree();
    [[nodiscard]] bool belongs(float x, float y) const;
    [[nodiscard]] bool belongs(const std::pair<int, int> &point) const;

    QuadTree * getLeafContainingPoint(const std::pair<int, int> &point);
    std::pair<int, int> getNearestNeighbour(const std::pair<int, int> &point);
private:

    QuadTree * parent = nullptr;

    bool isLeaf = true;
    QuadTree * NW;
    QuadTree * NE;
    QuadTree * SW;
    QuadTree * SE;

    void getChildrenPoints(std::unordered_map<float, std::unordered_set<float>> &pointsSet);
    QuadTree(float posX, float posY, float sizeX, float sizeY, int pointsLimit, int levelLimit, int level,
             QuadTree *parent);
};


#endif //NEURALNETLIBRARY_QUADTREE_H
