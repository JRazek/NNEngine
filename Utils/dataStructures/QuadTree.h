//
// Created by jrazek on 09.08.2021.
//

#ifndef NEURALNETLIBRARY_QUADTREE_H
#define NEURALNETLIBRARY_QUADTREE_H
#include <unordered_set>
#include <unordered_map>
#include <vector>

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

    std::unordered_map<float, std::unordered_set<float>> points;

    ~QuadTree();
    [[nodiscard]] bool belongs(float x, float y) const;
    [[nodiscard]] bool belongs(const std::pair<int, int> &point) const;

    std::pair<int, int> getNearestNeighbour(const std::pair<int, int> &point);
private:

    QuadTree * parent = nullptr;

    /**
     * 0 - parent
     * 1 - leaf
     */
    bool type;
    QuadTree * NW;
    QuadTree * NE;
    QuadTree * SW;
    QuadTree * SE;

    QuadTree(float posX, float posY, float sizeX, float sizeY, int pointsLimit, int levelLimit, int level,
             QuadTree *parent);
};


#endif //NEURALNETLIBRARY_QUADTREE_H
