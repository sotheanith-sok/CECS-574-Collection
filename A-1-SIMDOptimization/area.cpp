#include <iostream>
#include <immintrin.h>
#include <time.h>
#include <vector>
#include <ctime>
#include <math.h>

using namespace std;

/*
*Given two vectors of width and heights, calcualte the total area of all rectangles. Linear Operations
*/
float naive_operation(vector<float> width, vector<float> height)
{
    //Calculate area of each rectangle
    vector<float> areas;
    for (int i = 0; i < width.size(); i++)
    {
        areas.push_back(width[i] * height[i]);
    }

    //Calculate the total area of all rectangles
    float total = 0;
    for (int i = 0; i < areas.size(); i++)
    {
        total += areas[i];
    }

    return total;
}

/*
*Given two vectors of width and heights, calcualte the total area of all rectangles. SIMD Operations
*/
float simd_operation(vector<float> width, vector<float> height)
{
    //Calculate area of each rectangle
    __m256 areas = _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (int i = 0; i < width.size() / 8; ++i)
    {
        __m256 w = _mm256_loadu_ps(width.data() + i * 8);
        __m256 h = _mm256_loadu_ps(height.data() + i * 8);
        __m256 a = _mm256_mul_ps(w, h);
        areas = _mm256_add_ps(areas, a);
    }

    //Calculate the total area of all rectangles
    __m256 temp = _mm256_hadd_ps(areas, areas);
    temp = _mm256_hadd_ps(temp, temp);
    __m128 sum_high = _mm256_extractf128_ps(temp, 1);
    __m128 total = _mm_add_ps(sum_high, _mm256_castps256_ps128(temp));

    return ((float *)&total)[0];
}

/*Start of main program*/
int main()
{
    //Generate some random values for width and height.
    vector<float> width;
    vector<float> height;
    srand(time(0));
    int n = 1000000;
    for (int i = 0; i < n-(n%8); i++)
    {
        width.push_back((((float)rand())/RAND_MAX)*100);
        height.push_back((((float)rand())/RAND_MAX)*100);
    }
    
    //Start testing  
    int m = 1000; //number of iteration

    //Normal operations
    clock_t startTime = clock();
    float naive_result = 0;
    for (int i = 0; i < m; i++)
    {
        naive_result += naive_operation(width, height) / (float)(m);
    }
    double naive_time = (clock() - startTime) / (double)(CLOCKS_PER_SEC);

    //SIMD operations
    startTime = clock();
    float simd_result = 0;
    for (int i = 0; i < m; i++)
    {
        simd_result += simd_operation(width, height) / (float)(m);
    }
    double simd_time = (clock() - startTime) / (double)(CLOCKS_PER_SEC);

    //Print results
    cout << "Naive operations result: " << naive_result << endl;
    cout << "256 bit vectorize operations result: " << simd_result << endl;

    cout << "Time elpased is " << naive_time << " seconds for naive operations" << endl;
    cout << "Time elpased is " << simd_time << " seconds for 256 bit vectorize operations" << endl;

    cout << "Speedup: " << naive_time / simd_time << " x " << endl;

    return 0;
}
