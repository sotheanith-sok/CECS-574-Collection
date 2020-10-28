#include <immintrin.h>
#include <iostream>
using namespace std;
int sum_naive(int n, int *a)
{
    int sum = 0;
    for (int i = 0; i < n; i++)
        sum += a[i];
    return sum;
}

int sum_simd(int n, int *a)
{
    __m256i totals = _mm256_setzero_si256();
    for (int i = 0; i < n / 8; i++)
    {
        __m256i temp = _mm256_loadu_si256((__m256i *)(a + i * 8));

        totals = _mm256_add_epi16(totals, temp);
    }

    return ((int *)&totals)[0] + ((int *)&totals)[1] + ((int *)&totals)[2] + ((int *)&totals)[3] + ((int *)&totals)[4] + ((int *)&totals)[5] + ((int *)&totals)[6] + ((int *)&totals)[7];
}

int main(){
    int a [8]={1,2,3,4,5,6,7,8};

    cout<<"Naive results:"<< sum_naive(8,a)<<endl;
    cout<<"SIMD results:"<< sum_simd(8,a)<<endl;
    
}