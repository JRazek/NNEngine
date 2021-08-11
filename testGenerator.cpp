#include <bits/stdc++.h>
using namespace  std;

pair<float, float> nearestNeighbor(const pair<float, float> &point, vector<pair<float, float>> &points){

}

int main() {
    int seed = 0;
    srand(seed);

    const int pointsCount = 1e4;
    const int testsCount = 1e4;

    vector<pair<float, float>> points(pointsCount);

    for(int i = 0; i < pointsCount; i ++){
        points[i] = {rand() % 10000, rand() % 10000};
    }

    vector<pair<float, float>> tests(testsCount);
    for(int i = 0; i < testsCount; i ++){
        tests[i] = {rand() % 10000, rand() % 10000};
    }

    vector<pair<float, float>> answers(testsCount);

    for(int i = 0; i < testsCount; i ++){
        pair<float, float> ans = nearestNeighbor(answers[i], points);
    }



    return 0;
}
