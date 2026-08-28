#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace autotune {

struct RunConfig {
    int generation_threads = 8;
    int prompt_threads = 8;
    int logical_batch = 512;
    int micro_batch = 256;
    int context_size = 1024;
    int tokens_to_generate = 8;
};

struct MemoryStats {
    std::uint64_t model_bytes = 0;
    std::uint64_t context_bytes = 0;
    std::uint64_t compute_bytes = 0;

    [[nodiscard]] std::uint64_t total_bytes() const;
};

struct RunResult {
    int prompt_tokens = 0;
    int generated_tokens = 0;
    int actual_context = 0;
    int actual_logical_batch = 0;
    int actual_micro_batch = 0;
    double context_init_ms = 0.0;
    double prompt_time_ms = 0.0;
    double generation_time_ms = 0.0;
    MemoryStats memory;
    std::string output_text;

    [[nodiscard]] double prompt_tokens_per_second() const;
    [[nodiscard]] double generation_tokens_per_second() const;
    [[nodiscard]] double total_time_ms() const;
};

struct ModelInfo {
    std::filesystem::path path;
    std::string description;
    std::uint64_t parameter_count = 0;
    std::uint64_t file_bytes = 0;
    double load_time_ms = 0.0;
};

struct SystemMemoryInfo {
    std::uint64_t available_bytes = 0;
    std::uint64_t swap_total_bytes = 0;
    std::uint64_t swap_free_bytes = 0;

    [[nodiscard]] double swap_used_fraction() const;
};

class BackendRuntime {
public:
    explicit BackendRuntime(bool verbose_logging = false);
    ~BackendRuntime();

    BackendRuntime(const BackendRuntime &) = delete;
    BackendRuntime & operator=(const BackendRuntime &) = delete;
};

class Engine {
public:
    explicit Engine(const std::filesystem::path & model_path);
    ~Engine();

    Engine(Engine &&) noexcept;
    Engine & operator=(Engine &&) noexcept;
    Engine(const Engine &) = delete;
    Engine & operator=(const Engine &) = delete;

    [[nodiscard]] const ModelInfo & model_info() const;
    [[nodiscard]] std::vector<std::int32_t> tokenize_text(const std::string & text) const;
    [[nodiscard]] std::vector<std::int32_t> make_fixed_workload(int target_tokens) const;
    [[nodiscard]] RunResult run_text(
        const std::string & prompt,
        const RunConfig & config,
        bool stop_at_end_token = true,
        const std::atomic_bool * cancel = nullptr) const;
    [[nodiscard]] RunResult run_tokens(
        const std::vector<std::int32_t> & prompt_tokens,
        const RunConfig & config,
        bool capture_output = false,
        bool stop_at_end_token = false,
        const std::atomic_bool * cancel = nullptr) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct SweepOptions {
    int context_size = 1024;
    int logical_batch = 512;
    int long_prompt_tokens = 384;
    int generation_tokens = 8;
    int repetitions = 1;
    int finalist_repetitions = 3;
    bool smoke = false;
};

struct BenchmarkRow {
    std::string phase;
    RunConfig config;
    int target_prompt_tokens = 0;
    int repetitions = 0;
    RunResult result;
    double prompt_variation_percent = 0.0;
    double generation_variation_percent = 0.0;
    double tradeoff_score = -1.0;
    bool succeeded = true;
    std::string error;
};

struct Recommendation {
    std::string name;
    std::string reason;
    RunConfig config;
    double score = -1.0;
};

struct SweepReport {
    ModelInfo model;
    SystemMemoryInfo system_memory;
    std::vector<std::string> warnings;
    std::vector<BenchmarkRow> rows;
    std::vector<Recommendation> recommendations;
    RunConfig selected_config;
    bool cancelled = false;
};

struct ProgressUpdate {
    std::string phase;
    std::string message;
    RunConfig config;
    int repetition = 0;
    int repetitions = 0;
    int completed_steps = 0;
    int total_steps = 0;
    bool has_completed_row = false;
    BenchmarkRow completed_row;
};

using ProgressCallback = std::function<void(const ProgressUpdate &)>;

[[nodiscard]] SweepReport run_sweep(
    const Engine & engine,
    const SweepOptions & options,
    const ProgressCallback & progress = {},
    const std::atomic_bool * cancel = nullptr);

[[nodiscard]] SystemMemoryInfo read_system_memory();
[[nodiscard]] double bytes_to_mib(std::uint64_t bytes);

}  // namespace autotune
