#include <iostream>
#include <immintrin.h>
#include <time.h>
#include <vector>
#include <ctime>
#include <math.h>
#include <chrono>

using namespace std;

/*
*Given three vectors of m, x, b. Calculate the sum of the resulting y from m, x, and b using y=mx+b
*/
float naive_operation(vector<float> &m, vector<float> &x, vector<float> &b)
{
    //Calculate sum of y given m, x, and b
    float total = 0;

    for (int i = 0; i < m.size(); i++)
    {
        total += m[i] * x[i] + b[i];
    }

    return total;
}

/*
*Given three vectors of m, x, b. Calculate the sum of the resulting y from m, x, and b using y=mx+b. SIMD Operations
*/
float simd_operation(vector<float> &m, vector<float> &x, vector<float> &b)
{
    __m256 totals = _mm256_setzero_ps();

    for (int i = 0; i < m.size() / 8; ++i)
    {
        __m256 p_m = _mm256_loadu_ps(m.data() + i * 8);
        __m256 p_x = _mm256_loadu_ps(x.data() + i * 8);
        __m256 p_b = _mm256_loadu_ps(b.data() + i * 8);

        __m256 temp = _mm256_mul_ps(p_m, p_x);
        temp = _mm256_add_ps(temp, p_b);
        totals = _mm256_add_ps(totals, temp);
    }

    return ((float *)&totals)[0] + ((float *)&totals)[1] + ((float *)&totals)[2] + ((float *)&totals)[3] + ((float *)&totals)[4] + ((float *)&totals)[5] + ((float *)&totals)[6] + ((float *)&totals)[7];
}

/*Start of main program*/
int main()
{
    //Generate some random values for m, x , and b
    vector<float> m;
    vector<float> x;
    vector<float> b;

    srand(time(0)); //Set seed for rand

    int n = 100000; //numbers of m,x, and b

    for (int i = 0; i < n - (n % 8); i++)
    {
        m.push_back((((float)rand()) / RAND_MAX) * 100 - 50);
        x.push_back((((float)rand()) / RAND_MAX) * 100 - 50);
        b.push_back((((float)rand()) / RAND_MAX) * 100 - 50);
    }

    //Start testing
    int iter = 100000; //number of iteration to run each functions

    //Normal operations
    float naive_result = 0;

    chrono::steady_clock::time_point start = chrono::steady_clock::now();

    for (int i = 0; i < iter; i++)
    {
        naive_result = naive_operation(m, x, b);
    }

    chrono::steady_clock::time_point finish = chrono::steady_clock::now();
    double naive_time = chrono::duration_cast<std::chrono::duration<double>>(finish - start).count() / iter;

    //SIMD operations
    float simd_result = 0;
    start = chrono::steady_clock::now();
    for (int i = 0; i < iter; i++)
    {
        simd_result = simd_operation(m, x, b);
    }
    finish = chrono::steady_clock::now();
    double simd_time = chrono::duration_cast<std::chrono::duration<double>>(finish - start).count() / iter;

    //Print results

    cout << "Number of iteration : " << iter << endl;

    cout << "Naive operations result: " << naive_result << endl;
    cout << "256 bit vectorize operations result: " << simd_result << endl;

    cout << "Averaged time elpased is " << naive_time << " seconds for naive operations" << endl;
    cout << "Averaged Time elpased is " << simd_time << " seconds for 256 bit vectorize operations" << endl;

    cout << "Speedup: " << naive_time / simd_time << " x " << endl;

    return 0;
}
