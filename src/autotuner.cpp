#include "autotuner.hpp"

#include "llama-ext.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>

namespace autotune {
namespace {

using Clock = std::chrono::steady_clock;

bool g_verbose_logging = false;

class BenchmarkCancelled final : public std::runtime_error {
public:
    BenchmarkCancelled() : std::runtime_error("benchmark cancelled") {}
};

void log_callback(const ggml_log_level level, const char * text, void *) {
    if (g_verbose_logging || level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

double elapsed_ms(const Clock::time_point start, const Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct ModelDeleter {
    void operator()(llama_model * model) const {
        if (model != nullptr) {
            llama_model_free(model);
        }
    }
};

struct ContextDeleter {
    void operator()(llama_context * context) const {
        if (context != nullptr) {
            llama_free(context);
        }
    }
};

struct SamplerDeleter {
    void operator()(llama_sampler * sampler) const {
        if (sampler != nullptr) {
            llama_sampler_free(sampler);
        }
    }
};

using ModelPtr = std::unique_ptr<llama_model, ModelDeleter>;
using ContextPtr = std::unique_ptr<llama_context, ContextDeleter>;
using SamplerPtr = std::unique_ptr<llama_sampler, SamplerDeleter>;

MemoryStats query_memory(const llama_context * context) {
    MemoryStats stats;
    const llama_memory_breakdown breakdown = llama_get_memory_breakdown(context);
    for (const auto & entry : breakdown) {
        stats.model_bytes += entry.second.model;
        stats.context_bytes += entry.second.context;
        stats.compute_bytes += entry.second.compute;
    }
    return stats;
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    const int count = llama_tokenize(
        vocab, text.c_str(), static_cast<std::int32_t>(text.size()), nullptr, 0, true, true);
    if (count >= 0) {
        throw std::runtime_error("llama.cpp returned an invalid tokenization size");
    }

    std::vector<llama_token> tokens(static_cast<std::size_t>(-count));
    const int written = llama_tokenize(
        vocab,
        text.c_str(),
        static_cast<std::int32_t>(text.size()),
        tokens.data(),
        static_cast<std::int32_t>(tokens.size()),
        true,
        true);
    if (written < 0) {
        throw std::runtime_error("llama.cpp failed to tokenize the prompt");
    }
    tokens.resize(static_cast<std::size_t>(written));
    return tokens;
}

std::string token_to_piece(const llama_vocab * vocab, const llama_token token) {
    std::vector<char> buffer(128);
    int written = llama_token_to_piece(
        vocab, token, buffer.data(), static_cast<std::int32_t>(buffer.size()), 0, true);
    if (written < 0) {
        buffer.resize(static_cast<std::size_t>(-written));
        written = llama_token_to_piece(
            vocab, token, buffer.data(), static_cast<std::int32_t>(buffer.size()), 0, true);
    }
    if (written < 0) {
        throw std::runtime_error("llama.cpp failed to decode a generated token");
    }
    return std::string(buffer.data(), static_cast<std::size_t>(written));
}

void validate_config(const RunConfig & config) {
    if (config.generation_threads <= 0 || config.prompt_threads <= 0 ||
        config.logical_batch <= 0 || config.micro_batch <= 0 || config.context_size <= 0 ||
        config.tokens_to_generate <= 0) {
        throw std::runtime_error("all benchmark configuration values must be positive");
    }
    if (config.micro_batch > config.logical_batch) {
        throw std::runtime_error("prompt micro-batch cannot exceed the logical batch size");
    }
}

bool is_cancelled(const std::atomic_bool * cancel) {
    return cancel != nullptr && cancel->load();
}

bool abort_if_cancelled(void * data) {
    const auto * cancel = static_cast<const std::atomic_bool *>(data);
    return cancel != nullptr && cancel->load();
}

void check_decode_status(
    const int status,
    const std::string_view stage,
    const std::atomic_bool * cancel) {
    if (status == 0) {
        return;
    }
    if (status == 2 && is_cancelled(cancel)) {
        throw BenchmarkCancelled();
    }

    std::ostringstream message;
    message << "llama_decode failed during " << stage << " with status " << status;
    if (status == 1) {
        message << " (no KV-cache slot available)";
    } else if (status == 2) {
        message << " (operation aborted)";
    } else if (status == -1) {
        message << " (invalid input batch)";
    }
    throw std::runtime_error(message.str());
}

double median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 0) {
        return (values[middle - 1] + values[middle]) / 2.0;
    }
    return values[middle];
}

double coefficient_of_variation_percent(const std::vector<double> & values) {
    if (values.size() < 2) {
        return 0.0;
    }
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) /
                        static_cast<double>(values.size());
    if (mean <= 0.0) {
        return 0.0;
    }
    double squared_error = 0.0;
    for (const double value : values) {
        const double difference = value - mean;
        squared_error += difference * difference;
    }
    const double deviation = std::sqrt(squared_error / static_cast<double>(values.size() - 1));
    return 100.0 * deviation / mean;
}

std::vector<int> bounded_values(const std::vector<int> & values, const int maximum) {
    std::vector<int> result;
    for (const int value : values) {
        const int bounded = std::max(1, std::min(value, maximum));
        if (std::find(result.begin(), result.end(), bounded) == result.end()) {
            result.push_back(bounded);
        }
    }
    return result;
}

BenchmarkRow aggregate_case(
    const Engine & engine,
    const std::string & phase,
    const RunConfig & config,
    const std::vector<std::int32_t> & prompt_tokens,
    const int repetitions,
    const int completed,
    const int total,
    const ProgressCallback & progress,
    const std::atomic_bool * cancel) {
    std::vector<RunResult> trials;
    trials.reserve(static_cast<std::size_t>(repetitions));

    for (int repetition = 1; repetition <= repetitions; ++repetition) {
        if (is_cancelled(cancel)) {
            throw BenchmarkCancelled();
        }
        if (progress) {
            ProgressUpdate update;
            update.phase = phase;
            update.message = "running";
            update.config = config;
            update.repetition = repetition;
            update.repetitions = repetitions;
            update.completed_steps = completed;
            update.total_steps = total;
            progress(update);
        }
        trials.push_back(engine.run_tokens(prompt_tokens, config, false, false, cancel));
    }

    std::vector<double> context_times;
    std::vector<double> prompt_times;
    std::vector<double> generation_times;
    std::vector<double> prompt_rates;
    std::vector<double> generation_rates;
    context_times.reserve(trials.size());
    prompt_times.reserve(trials.size());
    generation_times.reserve(trials.size());
    prompt_rates.reserve(trials.size());
    generation_rates.reserve(trials.size());

    MemoryStats largest_memory;
    for (const RunResult & trial : trials) {
        context_times.push_back(trial.context_init_ms);
        prompt_times.push_back(trial.prompt_time_ms);
        generation_times.push_back(trial.generation_time_ms);
        prompt_rates.push_back(trial.prompt_tokens_per_second());
        generation_rates.push_back(trial.generation_tokens_per_second());
        if (trial.memory.total_bytes() > largest_memory.total_bytes()) {
            largest_memory = trial.memory;
        }
    }

    RunResult result = trials.front();
    result.context_init_ms = median(context_times);
    result.prompt_time_ms = median(prompt_times);
    result.generation_time_ms = median(generation_times);
    result.memory = largest_memory;
    result.output_text.clear();

    BenchmarkRow row;
    row.phase = phase;
    row.config = config;
    row.target_prompt_tokens = static_cast<int>(prompt_tokens.size());
    row.repetitions = repetitions;
    row.result = std::move(result);
    row.prompt_variation_percent = coefficient_of_variation_percent(prompt_rates);
    row.generation_variation_percent = coefficient_of_variation_percent(generation_rates);
    return row;
}

BenchmarkRow failed_case(
    const std::string & phase,
    const RunConfig & config,
    const int prompt_tokens,
    const std::string & error) {
    BenchmarkRow row;
    row.phase = phase;
    row.config = config;
    row.target_prompt_tokens = prompt_tokens;
    row.result.prompt_tokens = prompt_tokens;
    row.succeeded = false;
    row.error = error;
    return row;
}

void emit_completed_row(
    const ProgressCallback & progress,
    const BenchmarkRow & row,
    const int completed,
    const int total) {
    if (!progress) {
        return;
    }
    ProgressUpdate update;
    update.phase = row.phase;
    update.message = row.succeeded ? "completed" : "failed";
    update.config = row.config;
    update.completed_steps = completed;
    update.total_steps = total;
    update.has_completed_row = true;
    update.completed_row = row;
    progress(update);
}

bool same_config(const RunConfig & left, const RunConfig & right) {
    return left.generation_threads == right.generation_threads &&
           left.prompt_threads == right.prompt_threads &&
           left.logical_batch == right.logical_batch && left.micro_batch == right.micro_batch &&
           left.context_size == right.context_size &&
           left.tokens_to_generate == right.tokens_to_generate;
}

double workload_time_ms(const BenchmarkRow & row) {
    return row.result.prompt_time_ms + row.result.generation_time_ms;
}

const BenchmarkRow & fastest_generation(const std::vector<BenchmarkRow> & rows) {
    return *std::max_element(rows.begin(), rows.end(), [](const auto & left, const auto & right) {
        return left.result.generation_tokens_per_second() <
               right.result.generation_tokens_per_second();
    });
}

const BenchmarkRow & fastest_prompt(const std::vector<BenchmarkRow> & rows) {
    return *std::max_element(rows.begin(), rows.end(), [](const auto & left, const auto & right) {
        return left.result.prompt_tokens_per_second() < right.result.prompt_tokens_per_second();
    });
}

std::uint64_t parse_meminfo_kib(const std::string & line) {
    std::istringstream input(line);
    std::string label;
    std::uint64_t kib = 0;
    input >> label >> kib;
    return kib;
}

}  // namespace

std::uint64_t MemoryStats::total_bytes() const {
    return model_bytes + context_bytes + compute_bytes;
}

double RunResult::prompt_tokens_per_second() const {
    return prompt_time_ms > 0.0 ? 1000.0 * prompt_tokens / prompt_time_ms : 0.0;
}

double RunResult::generation_tokens_per_second() const {
    return generation_time_ms > 0.0 ? 1000.0 * generated_tokens / generation_time_ms : 0.0;
}

double RunResult::total_time_ms() const {
    return context_init_ms + prompt_time_ms + generation_time_ms;
}

double SystemMemoryInfo::swap_used_fraction() const {
    if (swap_total_bytes == 0) {
        return 0.0;
    }
    return static_cast<double>(swap_total_bytes - swap_free_bytes) /
           static_cast<double>(swap_total_bytes);
}

BackendRuntime::BackendRuntime(const bool verbose_logging) {
    g_verbose_logging = verbose_logging;
    llama_log_set(log_callback, nullptr);
    llama_backend_init();
}

BackendRuntime::~BackendRuntime() {
    llama_backend_free();
}

struct Engine::Impl {
    explicit Impl(const std::filesystem::path & model_path) {
        if (!std::filesystem::is_regular_file(model_path)) {
            throw std::runtime_error("model file does not exist: " + model_path.string());
        }

        const auto load_start = Clock::now();
        llama_model_params params = llama_model_default_params();
        params.n_gpu_layers = 0;
        model.reset(llama_model_load_from_file(model_path.string().c_str(), params));
        const auto load_end = Clock::now();
        if (!model) {
            throw std::runtime_error("llama.cpp failed to load the GGUF model");
        }
        if (llama_model_has_encoder(model.get())) {
            throw std::runtime_error(
                "encoder-decoder GGUF models are not supported; choose a decoder-only LLM");
        }
        if (!llama_model_has_decoder(model.get())) {
            throw std::runtime_error("this GGUF has no decoder and cannot generate text");
        }

        info.path = model_path;
        info.file_bytes = std::filesystem::file_size(model_path);
        info.load_time_ms = elapsed_ms(load_start, load_end);
        info.parameter_count = llama_model_n_params(model.get());

        std::vector<char> description(256);
        int written = llama_model_desc(model.get(), description.data(), description.size());
        if (written >= static_cast<int>(description.size())) {
            description.resize(static_cast<std::size_t>(written) + 1);
            written = llama_model_desc(model.get(), description.data(), description.size());
        }
        if (written > 0) {
            info.description.assign(description.data(), static_cast<std::size_t>(written));
        } else {
            info.description = "GGUF language model";
        }
    }

    ModelPtr model;
    ModelInfo info;
};

Engine::Engine(const std::filesystem::path & model_path) : impl_(std::make_unique<Impl>(model_path)) {}

Engine::~Engine() = default;
Engine::Engine(Engine &&) noexcept = default;
Engine & Engine::operator=(Engine &&) noexcept = default;

const ModelInfo & Engine::model_info() const {
    return impl_->info;
}

std::vector<std::int32_t> Engine::tokenize_text(const std::string & text) const {
    const llama_vocab * vocab = llama_model_get_vocab(impl_->model.get());
    const std::vector<llama_token> tokens = tokenize(vocab, text);
    return std::vector<std::int32_t>(tokens.begin(), tokens.end());
}

std::vector<std::int32_t> Engine::make_fixed_workload(const int target_tokens) const {
    if (target_tokens <= 0) {
        throw std::runtime_error("target prompt token count must be positive");
    }

    static const std::string passage =
        "Local language models can keep private data on the user's computer. "
        "Runtime settings change how quickly prompts are processed and how quickly answers "
        "are generated. A fair benchmark holds the model, prompt, and output length constant "
        "while measuring one configuration at a time. ";

    std::string text;
    std::vector<std::int32_t> tokens;
    while (static_cast<int>(tokens.size()) < target_tokens) {
        text += passage;
        tokens = tokenize_text(text);
    }
    tokens.resize(static_cast<std::size_t>(target_tokens));
    return tokens;
}

RunResult Engine::run_text(
    const std::string & prompt,
    const RunConfig & config,
    const bool stop_at_end_token,
    const std::atomic_bool * cancel) const {
    return run_tokens(tokenize_text(prompt), config, true, stop_at_end_token, cancel);
}

RunResult Engine::run_tokens(
    const std::vector<std::int32_t> & prompt_tokens,
    const RunConfig & config,
    const bool capture_output,
    const bool stop_at_end_token,
    const std::atomic_bool * cancel) const {
    validate_config(config);
    if (is_cancelled(cancel)) {
        throw BenchmarkCancelled();
    }
    if (prompt_tokens.empty()) {
        throw std::runtime_error("prompt produced no tokens");
    }

    const int required_context =
        static_cast<int>(prompt_tokens.size()) + config.tokens_to_generate + 8;
    const int requested_context = std::max(config.context_size, required_context);

    llama_context_params params = llama_context_default_params();
    params.n_ctx = static_cast<std::uint32_t>(requested_context);
    params.n_batch = static_cast<std::uint32_t>(config.logical_batch);
    params.n_ubatch = static_cast<std::uint32_t>(config.micro_batch);
    params.n_threads = config.generation_threads;
    params.n_threads_batch = config.prompt_threads;
    if (cancel != nullptr) {
        params.abort_callback = abort_if_cancelled;
        params.abort_callback_data = const_cast<std::atomic_bool *>(cancel);
    }

    const auto context_start = Clock::now();
    ContextPtr context(llama_init_from_model(impl_->model.get(), params));
    const auto context_end = Clock::now();
    if (!context) {
        if (is_cancelled(cancel)) {
            throw BenchmarkCancelled();
        }
        throw std::runtime_error(
            "llama.cpp could not allocate the inference context; reduce context or micro-batch");
    }

    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    SamplerPtr sampler(llama_sampler_chain_init(sampler_params));
    if (!sampler) {
        throw std::runtime_error("llama.cpp failed to create a sampler");
    }
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());

    std::vector<llama_token> native_prompt(prompt_tokens.begin(), prompt_tokens.end());
    const int actual_batch = static_cast<int>(llama_n_batch(context.get()));
    const auto prompt_start = Clock::now();
    std::size_t offset = 0;
    while (offset < native_prompt.size()) {
        const std::size_t remaining = native_prompt.size() - offset;
        const int chunk_size = static_cast<int>(
            std::min<std::size_t>(remaining, static_cast<std::size_t>(actual_batch)));
        llama_batch batch = llama_batch_get_one(native_prompt.data() + offset, chunk_size);
        check_decode_status(llama_decode(context.get(), batch), "prompt processing", cancel);
        offset += static_cast<std::size_t>(chunk_size);
    }
    llama_synchronize(context.get());
    const auto prompt_end = Clock::now();

    const llama_vocab * vocab = llama_model_get_vocab(impl_->model.get());
    std::vector<llama_token> generated_ids;
    if (capture_output) {
        generated_ids.reserve(static_cast<std::size_t>(config.tokens_to_generate));
    }
    int generated_tokens = 0;
    const auto generation_start = Clock::now();
    for (int i = 0; i < config.tokens_to_generate; ++i) {
        if (is_cancelled(cancel)) {
            throw BenchmarkCancelled();
        }
        const llama_token token = llama_sampler_sample(sampler.get(), context.get(), -1);
        if (stop_at_end_token && llama_vocab_is_eog(vocab, token)) {
            break;
        }
        if (capture_output) {
            generated_ids.push_back(token);
        }
        ++generated_tokens;

        llama_token next_input = token;
        llama_batch batch = llama_batch_get_one(&next_input, 1);
        check_decode_status(llama_decode(context.get(), batch), "token generation", cancel);
    }
    llama_synchronize(context.get());
    const auto generation_end = Clock::now();

    std::string output;
    for (const llama_token token : generated_ids) {
        output += token_to_piece(vocab, token);
    }

    RunResult result;
    result.prompt_tokens = static_cast<int>(prompt_tokens.size());
    result.generated_tokens = generated_tokens;
    result.actual_context = static_cast<int>(llama_n_ctx_seq(context.get()));
    result.actual_logical_batch = actual_batch;
    result.actual_micro_batch = static_cast<int>(llama_n_ubatch(context.get()));
    result.context_init_ms = elapsed_ms(context_start, context_end);
    result.prompt_time_ms = elapsed_ms(prompt_start, prompt_end);
    result.generation_time_ms = elapsed_ms(generation_start, generation_end);
    result.memory = query_memory(context.get());
    result.output_text = std::move(output);
    return result;
}

SystemMemoryInfo read_system_memory() {
    SystemMemoryInfo result;
    std::ifstream input("/proc/meminfo");
    std::string line;
    while (std::getline(input, line)) {
        if (line.rfind("MemAvailable:", 0) == 0) {
            result.available_bytes = parse_meminfo_kib(line) * 1024;
        } else if (line.rfind("SwapTotal:", 0) == 0) {
            result.swap_total_bytes = parse_meminfo_kib(line) * 1024;
        } else if (line.rfind("SwapFree:", 0) == 0) {
            result.swap_free_bytes = parse_meminfo_kib(line) * 1024;
        }
    }
    return result;
}

double bytes_to_mib(const std::uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

SweepReport run_sweep(
    const Engine & engine,
    const SweepOptions & options,
    const ProgressCallback & progress,
    const std::atomic_bool * cancel) {
    if (options.context_size <= 0 || options.logical_batch <= 0 ||
        options.long_prompt_tokens <= 0 || options.generation_tokens <= 0 ||
        options.repetitions <= 0 || options.finalist_repetitions <= 0) {
        throw std::runtime_error("all sweep options must be positive");
    }

    SweepReport report;
    report.model = engine.model_info();
    report.system_memory = read_system_memory();

    const std::uint64_t comfortable_memory = report.model.file_bytes + 1024ULL * 1024ULL * 1024ULL;
    if (report.system_memory.available_bytes > 0 &&
        report.system_memory.available_bytes < comfortable_memory) {
        report.warnings.push_back(
            "Available RAM is below model size plus 1 GiB; background apps may cause swapping.");
    }
    if (report.system_memory.swap_used_fraction() > 0.8) {
        report.warnings.push_back(
            "More than 80% of swap is already used; close memory-heavy apps for stable results.");
    }

    const unsigned int detected = std::thread::hardware_concurrency();
    const int maximum_threads = detected == 0 ? 16 : static_cast<int>(detected);
    const std::vector<int> thread_values = options.smoke
        ? bounded_values({2, 8}, maximum_threads)
        : bounded_values({2, 4, 8, 16}, maximum_threads);
    std::vector<int> micro_values = options.smoke
        ? std::vector<int>{64, 128}
        : std::vector<int>{64, 128, 256, 512};
    micro_values.erase(
        std::remove_if(micro_values.begin(), micro_values.end(), [&](const int value) {
            return value > options.logical_batch;
        }),
        micro_values.end());
    if (micro_values.empty()) {
        micro_values.push_back(options.logical_batch);
    }

    const int base_threads = std::min(8, maximum_threads);
    const int base_micro = std::min(256, options.logical_batch);
    const int short_prompt_count = std::min(64, options.long_prompt_tokens);
    const std::vector<std::int32_t> short_prompt =
        engine.make_fixed_workload(short_prompt_count);
    const std::vector<std::int32_t> long_prompt =
        engine.make_fixed_workload(options.long_prompt_tokens);

    std::mt19937 random(0x51A7U);
    std::vector<int> generation_order = thread_values;
    std::vector<int> prompt_order = thread_values;
    std::shuffle(generation_order.begin(), generation_order.end(), random);
    std::shuffle(prompt_order.begin(), prompt_order.end(), random);
    std::shuffle(micro_values.begin(), micro_values.end(), random);

    const int coordinate_steps = 1 + static_cast<int>(thread_values.size()) * 2 +
                                 static_cast<int>(micro_values.size());
    int total_steps = coordinate_steps + 12;
    int completed = 0;

    RunConfig warm_config;
    warm_config.generation_threads = base_threads;
    warm_config.prompt_threads = base_threads;
    warm_config.logical_batch = options.logical_batch;
    warm_config.micro_batch = std::min(64, options.logical_batch);
    warm_config.context_size = options.context_size;
    warm_config.tokens_to_generate = std::min(2, options.generation_tokens);
    if (progress) {
        ProgressUpdate update;
        update.phase = "Warm-up";
        update.message = "warming model";
        update.config = warm_config;
        update.repetition = 1;
        update.repetitions = 1;
        update.completed_steps = completed;
        update.total_steps = total_steps;
        progress(update);
    }
    if (is_cancelled(cancel)) {
        report.cancelled = true;
        return report;
    }
    try {
        static_cast<void>(engine.run_tokens(
            engine.make_fixed_workload(std::min(16, short_prompt_count)),
            warm_config,
            false,
            false,
            cancel));
    } catch (const BenchmarkCancelled &) {
        report.cancelled = true;
        return report;
    }
    ++completed;

    const auto measure = [&](const std::string & phase,
                             const RunConfig & config,
                             const std::vector<std::int32_t> & prompt_tokens,
                             const int repetitions) -> std::optional<BenchmarkRow> {
        try {
            BenchmarkRow row = aggregate_case(
                engine,
                phase,
                config,
                prompt_tokens,
                repetitions,
                completed,
                total_steps,
                progress,
                cancel);
            ++completed;
            report.rows.push_back(row);
            emit_completed_row(progress, row, completed, total_steps);
            return row;
        } catch (const BenchmarkCancelled &) {
            report.cancelled = true;
            return std::nullopt;
        } catch (const std::exception & error) {
            BenchmarkRow row = failed_case(
                phase, config, static_cast<int>(prompt_tokens.size()), error.what());
            ++completed;
            report.rows.push_back(row);
            report.warnings.push_back(
                phase + " candidate failed (TG " +
                std::to_string(config.generation_threads) + ", PP " +
                std::to_string(config.prompt_threads) + ", micro-batch " +
                std::to_string(config.micro_batch) + "): " + error.what());
            emit_completed_row(progress, row, completed, total_steps);
            return std::nullopt;
        }
    };

    std::vector<BenchmarkRow> generation_rows;
    for (const int generation_threads : generation_order) {
        RunConfig config;
        config.generation_threads = generation_threads;
        config.prompt_threads = base_threads;
        config.logical_batch = options.logical_batch;
        config.micro_batch = base_micro;
        config.context_size = options.context_size;
        config.tokens_to_generate = options.generation_tokens;
        const std::optional<BenchmarkRow> row =
            measure("Generation threads", config, short_prompt, options.repetitions);
        if (report.cancelled) {
            return report;
        }
        if (row) {
            generation_rows.push_back(*row);
        }
    }
    if (generation_rows.empty()) {
        throw std::runtime_error("all generation-thread candidates failed");
    }

    const BenchmarkRow generation_winner = fastest_generation(generation_rows);

    std::vector<BenchmarkRow> prompt_rows;
    for (const int prompt_threads : prompt_order) {
        RunConfig config = generation_winner.config;
        config.prompt_threads = prompt_threads;
        const std::optional<BenchmarkRow> row =
            measure("Prompt threads", config, long_prompt, options.repetitions);
        if (report.cancelled) {
            return report;
        }
        if (row) {
            prompt_rows.push_back(*row);
        }
    }
    if (prompt_rows.empty()) {
        throw std::runtime_error("all prompt-thread candidates failed");
    }

    const BenchmarkRow prompt_winner = fastest_prompt(prompt_rows);

    std::vector<BenchmarkRow> micro_rows;
    for (const int micro_batch : micro_values) {
        RunConfig config = prompt_winner.config;
        config.micro_batch = micro_batch;
        const std::optional<BenchmarkRow> row =
            measure("Prompt micro-batch", config, long_prompt, options.repetitions);
        if (report.cancelled) {
            return report;
        }
        if (row) {
            micro_rows.push_back(*row);
        }
    }
    if (micro_rows.empty()) {
        throw std::runtime_error("all prompt micro-batch candidates failed");
    }

    const BenchmarkRow prompt_micro_winner = fastest_prompt(micro_rows);
    const BenchmarkRow memory_coordinate = *std::min_element(
        micro_rows.begin(), micro_rows.end(), [](const auto & left, const auto & right) {
            return left.result.memory.total_bytes() < right.result.memory.total_bytes();
        });

    struct Finalist {
        std::string name;
        RunConfig config;
        std::optional<BenchmarkRow> short_result;
        std::optional<BenchmarkRow> long_result;
        double score = -1.0;
    };

    std::vector<Finalist> finalists;
    const auto add_finalist = [&](const std::string & name, RunConfig config) {
        config.context_size = options.context_size;
        config.logical_batch = options.logical_batch;
        config.tokens_to_generate = options.generation_tokens;
        for (const Finalist & existing : finalists) {
            if (same_config(existing.config, config)) {
                return;
            }
        }
        Finalist finalist;
        finalist.name = name;
        finalist.config = config;
        finalists.push_back(std::move(finalist));
    };

    RunConfig baseline = generation_winner.config;
    baseline.generation_threads = base_threads;
    baseline.prompt_threads = base_threads;
    baseline.micro_batch = base_micro;
    add_finalist("Baseline", baseline);
    add_finalist("Chat winner", generation_winner.config);
    add_finalist("Prompt winner", prompt_winner.config);
    add_finalist("Long-prompt winner", prompt_micro_winner.config);
    add_finalist("Memory saver", memory_coordinate.config);
    RunConfig low_thread = prompt_micro_winner.config;
    low_thread.generation_threads = std::min(4, maximum_threads);
    low_thread.prompt_threads = std::min(4, maximum_threads);
    low_thread.micro_batch = std::min(128, options.logical_batch);
    add_finalist("Low-thread preset", low_thread);

    completed += 2 * (6 - static_cast<int>(finalists.size()));
    std::shuffle(finalists.begin(), finalists.end(), random);
    const int finalist_repetitions =
        options.smoke ? 1 : std::max(options.repetitions, options.finalist_repetitions);

    for (Finalist & finalist : finalists) {
        finalist.short_result = measure(
            "Finalist short - " + finalist.name,
            finalist.config,
            short_prompt,
            finalist_repetitions);
        if (report.cancelled) {
            return report;
        }
        finalist.long_result = measure(
            "Finalist long - " + finalist.name,
            finalist.config,
            long_prompt,
            finalist_repetitions);
        if (report.cancelled) {
            return report;
        }
    }

    std::vector<Finalist *> valid_finalists;
    for (Finalist & finalist : finalists) {
        if (finalist.short_result && finalist.long_result) {
            valid_finalists.push_back(&finalist);
        }
    }
    if (valid_finalists.empty()) {
        throw std::runtime_error("all full-configuration finalists failed");
    }

    double best_short_ms = std::numeric_limits<double>::max();
    double best_long_ms = std::numeric_limits<double>::max();
    std::uint64_t lowest_memory = std::numeric_limits<std::uint64_t>::max();
    for (const Finalist * finalist : valid_finalists) {
        best_short_ms = std::min(best_short_ms, workload_time_ms(*finalist->short_result));
        best_long_ms = std::min(best_long_ms, workload_time_ms(*finalist->long_result));
        lowest_memory = std::min(
            lowest_memory,
            std::max(
                finalist->short_result->result.memory.total_bytes(),
                finalist->long_result->result.memory.total_bytes()));
    }

    for (Finalist * finalist : valid_finalists) {
        const double short_ratio = best_short_ms / workload_time_ms(*finalist->short_result);
        const double long_ratio = best_long_ms / workload_time_ms(*finalist->long_result);
        const std::uint64_t finalist_memory = std::max(
            finalist->short_result->result.memory.total_bytes(),
            finalist->long_result->result.memory.total_bytes());
        const double memory_ratio =
            static_cast<double>(lowest_memory) / static_cast<double>(finalist_memory);
        finalist->score =
            100.0 * std::pow(short_ratio, 0.40) * std::pow(long_ratio, 0.40) *
            std::pow(memory_ratio, 0.20);

        for (BenchmarkRow & row : report.rows) {
            if (same_config(row.config, finalist->config) &&
                row.phase.rfind("Finalist ", 0) == 0) {
                row.tradeoff_score = finalist->score;
            }
        }
    }

    const Finalist * fastest_chat = *std::min_element(
        valid_finalists.begin(), valid_finalists.end(), [](const auto * left, const auto * right) {
            return workload_time_ms(*left->short_result) < workload_time_ms(*right->short_result);
        });
    const Finalist * fastest_long = *std::min_element(
        valid_finalists.begin(), valid_finalists.end(), [](const auto * left, const auto * right) {
            return workload_time_ms(*left->long_result) < workload_time_ms(*right->long_result);
        });
    const Finalist * memory_winner = *std::min_element(
        valid_finalists.begin(), valid_finalists.end(), [](const auto * left, const auto * right) {
            const std::uint64_t left_memory = std::max(
                left->short_result->result.memory.total_bytes(),
                left->long_result->result.memory.total_bytes());
            const std::uint64_t right_memory = std::max(
                right->short_result->result.memory.total_bytes(),
                right->long_result->result.memory.total_bytes());
            return left_memory < right_memory;
        });
    const Finalist * balanced_winner = *std::max_element(
        valid_finalists.begin(), valid_finalists.end(), [](const auto * left, const auto * right) {
            return left->score < right->score;
        });

    const auto actual_config = [](const Finalist * finalist) {
        RunConfig config = finalist->config;
        config.context_size = finalist->short_result->result.actual_context;
        config.logical_batch = finalist->short_result->result.actual_logical_batch;
        config.micro_batch = finalist->short_result->result.actual_micro_batch;
        return config;
    };

    report.selected_config = actual_config(balanced_winner);
    report.recommendations.push_back(Recommendation{
        "Fastest chat",
        "Lowest validated short-prompt plus generation latency.",
        actual_config(fastest_chat),
        workload_time_ms(*fastest_chat->short_result),
    });
    report.recommendations.push_back(Recommendation{
        "Long documents",
        "Lowest validated long-prompt plus generation latency.",
        actual_config(fastest_long),
        workload_time_ms(*fastest_long->long_result),
    });
    report.recommendations.push_back(Recommendation{
        "Memory saver",
        "Smallest validated llama.cpp model, context, and compute-buffer allocation.",
        actual_config(memory_winner),
        bytes_to_mib(std::max(
            memory_winner->short_result->result.memory.total_bytes(),
            memory_winner->long_result->result.memory.total_bytes())),
    });
    report.recommendations.push_back(Recommendation{
        "Balanced preset",
        "Weighted geometric mean: 40% short latency, 40% long latency, and 20% buffer efficiency.",
        actual_config(balanced_winner),
        balanced_winner->score,
    });

    if (progress) {
        ProgressUpdate update;
        update.phase = "Complete";
        update.message = "benchmark complete";
        update.config = report.selected_config;
        update.completed_steps = completed;
        update.total_steps = total_steps;
        progress(update);
    }
    return report;
}

}  // namespace autotune
