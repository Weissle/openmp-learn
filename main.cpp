#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <omp.h>
#include <thread>

using namespace std;
template <typename F> void stimer(F f, const char *info) {
    auto start = omp_get_wtime();
    cout << info << " start:" << endl;
    f();
    cout << info << " end. Time: " << omp_get_wtime() - start << endl;
}
#define STIMER(F) stimer(F, #F)

void hello() {
#pragma omp parallel
    {
        int ID = omp_get_thread_num();
        printf("hello %d/%d\n", ID, omp_get_num_threads());
    }
}

void sin_array_para() {
    const int size = 2 << 20;
    auto sinTable = make_unique<double[]>(size);
#pragma omp parallel for
    for (int n = 0; n < size; ++n)
        sinTable[n] = std::sin(2 * M_PI * n / size);
}

void sin_array_simd() {
    const int size = 2 << 20;
    auto sinTable = make_unique<double[]>(size);
#pragma omp for simd
    for (int n = 0; n < size; ++n)
        sinTable[n] = std::sin(2 * M_PI * n / size);
}

void sin_array_single() {
    const int size = 2 << 20;
    auto sinTable = make_unique<double[]>(size);
    for (int n = 0; n < size; ++n)
        sinTable[n] = std::sin(2 * M_PI * n / size);
}

void variable_atomic_test() {
    int pre = 5;
    int finnal = 0;
// #pragma omp parallel firstprivate(pre) num_threads(4)
#pragma omp parallel num_threads(4)
    {
        for (size_t i = 0; i < 10; i++) {
            pre += i;
        }
        // #pragma omp atomic update
        finnal += pre;
    }
    printf("%d\n", finnal);
}

void partial_ordered() {
#pragma omp parallel for ordered // schedule(dynamic)
    for (size_t i = 0; i < omp_get_num_threads(); i++) {
        printf("Thread %d starts.\n", omp_get_thread_num());
        std::this_thread::sleep_for(std::chrono::seconds(1));
        printf("Thread %d wakes up.\n", omp_get_thread_num());
#pragma omp ordered
        printf("Thread %d is done.\n", omp_get_thread_num());
    }
}

void loop_collapse() {
    constexpr int X = 30;
    int arr[X][X] = {};
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < X; ++i) {
        for (int j = 0; j < X; ++j)
            arr[i][j] = omp_get_thread_num();
    }

    for (int i = 0; i < X; ++i) {
        for (int j = 0; j < X; ++j)
            printf("%d ", arr[i][j]);
        printf("\n");
    }
}

void reduction() {
    long long sum = -1;
#pragma omp parallel for reduction(max : sum)
    for (int i = 0; i < 1000; ++i)
        sum += i;
    printf("%lld\n", sum);
    // printf("%lld\n", (0ll + 999) * 1000 / 2);
}

int main() {
    // STIMER(hello);
    // STIMER(sin_array_single);
    // STIMER(sin_array_para);
    // STIMER(sin_array_simd);
    // STIMER(variable_atomic_test);
    // STIMER(partial_ordered);
    // STIMER(loop_collapse);
    STIMER(reduction);
    return 0;
}