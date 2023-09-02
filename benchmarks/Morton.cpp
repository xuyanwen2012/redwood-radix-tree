// #include "Morton.hpp"

// #include <benchmark/benchmark.h>

// static void BM_LibMortonPointToCode(benchmark::State& state) {
//   float x = 1.0f;
//   float y = 1.0f;
//   float z = 1.0f;

//   for (auto _ : state) {
//     for (int i = 0; i < 1024 * 10; ++i) {
//       auto result = PointToCode(x, y, z);
//       benchmark::DoNotOptimize(result);
//     }
//   }
// }

// static void BM_MyPointToCode(benchmark::State& state) {
//   for (auto _ : state) {
//     for (int i = 0; i < 1024 * 10; ++i) {
//       auto result = MyVersionPointToCode(1.0f, 1.0f, 1.0f);
//       benchmark::DoNotOptimize(result);
//     }
//   }
// }

// BENCHMARK(BM_LibMortonPointToCode);
// BENCHMARK(BM_MyPointToCode);

// BENCHMARK_MAIN();
