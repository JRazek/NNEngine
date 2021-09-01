#include <bits/stdc++.h>
using namespace  std;

pair<double, double> nearestNeighbor(const pair<double, double> &point, vector<pair<double, double>> &points){
    double best = INFINITY;
    pair<double, double> bestPair;
    for(auto p : points){
        double res = pow(point.first - p.first, 2) + pow(point.second - p.second, 2);
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

    const int pointsCount = 100000;
    const int testsCount = 10000;


    const int maxX = 1000, maxY = 1000;

    vector<pair<double, double>> points(pointsCount);

    for(int i = 0; i < pointsCount; i ++){
        points[i] = {rand() % maxX, rand() % maxY};
    }

    vector<pair<double, double>> tests(testsCount);
    for(int i = 0; i < testsCount; i ++){
        tests[i] = {rand() % maxX, rand() % maxY};
    }

    vector<pair<double, double>> answers(testsCount);

    for(int i = 0; i < testsCount; i ++){
        pair<double, double> ans = nearestNeighbor(tests[i], points);
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
