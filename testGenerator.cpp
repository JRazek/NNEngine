#include <bits/stdc++.h>
using namespace  std;

pair<float, float> nearestNeighbor(const pair<float, float> &point, vector<pair<float, float>> &points){
    float best = INFINITY;
    pair<float, float> bestPair;
    for(auto p : points){
        float res = pow(point.first - p.first, 2) + pow(point.second - p.second, 2);
        if(res < best){
            best = res;
            bestPair = p;
        }
    }
    return bestPair;
}

int main() {
    int seed = 4;
    srand(seed);

    const int pointsCount = 10;
    const int testsCount = 10000;


    const int maxX = 200, maxY = 200;

    vector<pair<float, float>> points(pointsCount);

    for(int i = 0; i < pointsCount; i ++){
        points[i] = {rand() % maxX, rand() % maxY};
    }

    vector<pair<float, float>> tests(testsCount);
    for(int i = 0; i < testsCount; i ++){
        tests[i] = {rand() % maxX, rand() % maxY};
    }

    vector<pair<float, float>> answers(testsCount);

    for(int i = 0; i < testsCount; i ++){
        pair<float, float> ans = nearestNeighbor(tests[i], points);
        answers[i] = ans;
    }

    fstream outputFile("test.in", ios::out);

    outputFile << pointsCount << "\n";
    outputFile << testsCount << "\n";
    for(auto p : points){
        outputFile << p.first << " " << p.second << "\n";
    }

    for(int i = 0; i < testsCount; i ++){
        auto p = tests[i];
        auto ans = answers[i];
        outputFile << p.first << " " << p.second << " " << ans.first << " " << ans.second <<  "\n";
    }

    return 0;
}
