#include "autotuner.hpp"
#ifdef AUTOTUNE_GUI_AVAILABLE
#include "gui.hpp"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

namespace {

constexpr const char * kDefaultModel =
    "/usr/share/ollama/.ollama/models/blobs/"
    "sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff";

enum class Mode {
#ifdef AUTOTUNE_GUI_AVAILABLE
    Gui,
#endif
    Single,
    Sweep,
};

struct CliOptions {
#ifdef AUTOTUNE_GUI_AVAILABLE
    Mode mode = Mode::Gui;
#else
    Mode mode = Mode::Single;
#endif
    std::filesystem::path model_path = kDefaultModel;
    std::string prompt = "Write one short sentence explaining why local AI can be useful.";
    autotune::RunConfig single_config;
    autotune::SweepOptions sweep;
    bool verbose = false;
    bool gui_auto_run = false;
    bool gui_smoke = false;
    bool gui_exit_after_run = false;
    int cancel_after_ms = 0;
};

int parse_positive_int(const std::string_view option, const char * value) {
    try {
        const std::string text(value);
        std::size_t consumed = 0;
        const int parsed = std::stoi(text, &consumed);
        if (parsed <= 0 || consumed != text.size()) {
            throw std::invalid_argument("not positive");
        }
        return parsed;
    } catch (...) {
        throw std::runtime_error(std::string(option) + " expects a positive integer");
    }
}

void print_usage(const char * executable) {
    std::cout
        << "Local LLM Hardware Auto-Tuner\n\n"
        << "Usage: " << executable
#ifdef AUTOTUNE_GUI_AVAILABLE
        << " [--gui | --single | --sweep | --smoke] [options]\n\n"
#else
        << " [--single | --sweep | --smoke] [options]\n\n"
#endif
        << "Modes:\n"
#ifdef AUTOTUNE_GUI_AVAILABLE
        << "  --gui                    Open the live dashboard (default)\n"
        << "  --gui-smoke              Auto-run the GUI smoke profile\n"
#endif
        << "  --single                 Run one prompt/configuration\n"
        << "  --sweep                  Tune generation threads, prompt threads, and micro-batch\n"
        << "  --smoke                  Run a shortened sweep for build verification\n\n"
        << "Model option:\n"
        << "  --model PATH             GGUF model path (defaults to the local Llama 3.2 blob)\n"
        << "  --help                    Show this help\n\n"
        << "CLI workload options:\n"
        << "  --context N              Minimum context size (default: 1024)\n"
        << "  --tokens N               Tokens generated per measurement (default: 8)\n"
        << "  --verbose                Show llama.cpp informational logs\n"
#ifdef AUTOTUNE_GUI_AVAILABLE
        << "\nGUI automation options:\n"
        << "  --auto-run               Start the selected GUI benchmark immediately\n"
        << "  --exit-after-run         Close the GUI when an auto-run finishes\n"
#endif
        << "\n"
        << "Single-run options:\n"
        << "  --prompt TEXT            Prompt text\n"
        << "  --threads N              Generation threads (default: 8)\n"
        << "  --prompt-threads N       Prompt-processing threads (default: 8)\n"
        << "  --batch N                Logical batch size (default: 512)\n"
        << "  --micro-batch N          Physical prompt micro-batch (default: 256)\n\n"
        << "Sweep options:\n"
        << "  --prompt-tokens N        Long workload length (default: 384 tokens)\n"
        << "  --repetitions N          Exploratory repetitions per candidate (default: 1)\n"
        << "  --finalist-repetitions N Validated repetitions per full preset (default: 3)\n"
        << "  --cancel-after-ms N      Diagnostic: cancel a CLI sweep after N milliseconds\n";
}

CliOptions parse_arguments(const int argc, char ** argv) {
    CliOptions options;
    int selected_modes = 0;
    bool smoke_mode = false;
    [[maybe_unused]] bool cli_workload_option_seen = false;
    [[maybe_unused]] bool gui_automation_option_seen = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (argument == "--single") {
            options.mode = Mode::Single;
            ++selected_modes;
        } else if (argument == "--sweep") {
            options.mode = Mode::Sweep;
            ++selected_modes;
        } else if (argument == "--smoke") {
            options.mode = Mode::Sweep;
            smoke_mode = true;
            ++selected_modes;
#ifdef AUTOTUNE_GUI_AVAILABLE
        } else if (argument == "--gui") {
            options.mode = Mode::Gui;
            ++selected_modes;
        } else if (argument == "--gui-smoke") {
            options.mode = Mode::Gui;
            options.gui_auto_run = true;
            options.gui_smoke = true;
            ++selected_modes;
#endif
        }
    }
    if (selected_modes > 1) {
        throw std::runtime_error(
            "choose only one mode: --gui, --single, --sweep, --smoke, or --gui-smoke");
    }
    if (smoke_mode) {
        options.sweep.smoke = true;
        options.sweep.long_prompt_tokens = 16;
        options.sweep.generation_tokens = 2;
        options.sweep.finalist_repetitions = 1;
    }

    bool micro_batch_was_set = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        if (argument == "--help" || argument == "-h" || argument == "--single" ||
            argument == "--sweep" || argument == "--smoke") {
            continue;
        }
#ifdef AUTOTUNE_GUI_AVAILABLE
        if (argument == "--gui" || argument == "--gui-smoke") {
            continue;
        }
#else
        if (argument == "--gui" || argument == "--gui-smoke") {
            throw std::runtime_error("this binary was built with AUTOTUNE_BUILD_GUI=OFF");
        }
#endif
        if (argument == "--verbose") {
            options.verbose = true;
            cli_workload_option_seen = true;
            continue;
        }
#ifdef AUTOTUNE_GUI_AVAILABLE
        if (argument == "--auto-run") {
            options.gui_auto_run = true;
            gui_automation_option_seen = true;
            continue;
        }
        if (argument == "--exit-after-run") {
            options.gui_exit_after_run = true;
            gui_automation_option_seen = true;
            continue;
        }
#else
        if (argument == "--auto-run" || argument == "--exit-after-run") {
            throw std::runtime_error("this binary was built with AUTOTUNE_BUILD_GUI=OFF");
        }
#endif

        if (i + 1 >= argc) {
            throw std::runtime_error(std::string(argument) + " requires a value");
        }
        const char * value = argv[++i];
        if (argument == "--model") {
            options.model_path = value;
        } else if (argument == "--prompt") {
            options.prompt = value;
            cli_workload_option_seen = true;
        } else if (argument == "--threads") {
            options.single_config.generation_threads = parse_positive_int(argument, value);
            cli_workload_option_seen = true;
        } else if (argument == "--prompt-threads") {
            options.single_config.prompt_threads = parse_positive_int(argument, value);
            cli_workload_option_seen = true;
        } else if (argument == "--batch") {
            const int parsed = parse_positive_int(argument, value);
            options.single_config.logical_batch = parsed;
            options.sweep.logical_batch = parsed;
            cli_workload_option_seen = true;
            if (!micro_batch_was_set) {
                options.single_config.micro_batch =
                    std::min(options.single_config.micro_batch, parsed);
            }
        } else if (argument == "--micro-batch") {
            options.single_config.micro_batch = parse_positive_int(argument, value);
            micro_batch_was_set = true;
            cli_workload_option_seen = true;
        } else if (argument == "--context") {
            const int parsed = parse_positive_int(argument, value);
            options.single_config.context_size = parsed;
            options.sweep.context_size = parsed;
            cli_workload_option_seen = true;
        } else if (argument == "--tokens") {
            const int parsed = parse_positive_int(argument, value);
            options.single_config.tokens_to_generate = parsed;
            options.sweep.generation_tokens = parsed;
            cli_workload_option_seen = true;
        } else if (argument == "--prompt-tokens") {
            options.sweep.long_prompt_tokens = parse_positive_int(argument, value);
            cli_workload_option_seen = true;
        } else if (argument == "--repetitions") {
            options.sweep.repetitions = parse_positive_int(argument, value);
            cli_workload_option_seen = true;
        } else if (argument == "--finalist-repetitions") {
            options.sweep.finalist_repetitions = parse_positive_int(argument, value);
            cli_workload_option_seen = true;
        } else if (argument == "--cancel-after-ms") {
            options.cancel_after_ms = parse_positive_int(argument, value);
            cli_workload_option_seen = true;
        } else {
            throw std::runtime_error("unknown option: " + std::string(argument));
        }
    }
    if (options.mode == Mode::Single &&
        options.single_config.micro_batch > options.single_config.logical_batch) {
        throw std::runtime_error("--micro-batch cannot exceed --batch");
    }
#ifdef AUTOTUNE_GUI_AVAILABLE
    if (options.mode == Mode::Gui && cli_workload_option_seen) {
        throw std::runtime_error(
            "CLI workload options require --single, --sweep, or --smoke; GUI workloads are selected by profile");
    }
    if (options.mode != Mode::Gui && gui_automation_option_seen) {
        throw std::runtime_error("--auto-run and --exit-after-run require --gui or --gui-smoke");
    }
#endif
    return options;
}

void print_model_info(const autotune::ModelInfo & model) {
    std::cout << "Model:       " << model.description << '\n';
    std::cout << "Path:        " << model.path << '\n';
    std::cout << "Parameters:  " << std::fixed << std::setprecision(2)
              << static_cast<double>(model.parameter_count) / 1'000'000'000.0 << " billion\n";
    std::cout << "File size:   " << autotune::bytes_to_mib(model.file_bytes) << " MiB\n";
    std::cout << "Load time:   " << model.load_time_ms << " ms\n";
}

void print_single_result(
    const autotune::ModelInfo & model,
    const autotune::RunConfig & config,
    const autotune::RunResult & result) {
    std::cout << "\nLocal LLM single-run benchmark\n";
    print_model_info(model);
    std::cout << "Config:      generation threads " << config.generation_threads
              << ", prompt threads " << config.prompt_threads << ", logical batch "
              << result.actual_logical_batch << ", micro-batch " << result.actual_micro_batch
              << " (actual llama.cpp values)\n";
    std::cout << "Context init: " << result.context_init_ms << " ms\n";
    std::cout << "Prompt:      " << result.prompt_tokens << " tokens in "
              << result.prompt_time_ms << " ms (" << result.prompt_tokens_per_second()
              << " tok/s)\n";
    std::cout << "Generation:  " << result.generated_tokens << " tokens in "
              << result.generation_time_ms << " ms ("
              << result.generation_tokens_per_second() << " tok/s)\n";
    std::cout << "llama.cpp allocated buffers: "
              << autotune::bytes_to_mib(result.memory.total_bytes()) << " MiB [model "
              << autotune::bytes_to_mib(result.memory.model_bytes) << ", context "
              << autotune::bytes_to_mib(result.memory.context_bytes) << ", compute "
              << autotune::bytes_to_mib(result.memory.compute_bytes) << "]\n";
    std::cout << "Output:      " << result.output_text << "\n\n";
}

void print_sweep_table(const autotune::SweepReport & report) {
    std::cout << "\nMeasured candidates (median of each candidate's repetitions)\n";
    std::cout << std::left << std::setw(23) << "Phase" << std::right << std::setw(6) << "TG"
              << std::setw(6) << "PP" << std::setw(9) << "uBatch" << std::setw(8) << "PTok"
              << std::setw(12) << "PP tok/s" << std::setw(12) << "TG tok/s"
              << std::setw(13) << "Buffers MB" << std::setw(9) << "Score" << '\n';
    std::cout << std::string(98, '-') << '\n';
    for (const autotune::BenchmarkRow & row : report.rows) {
        if (!row.succeeded) {
            std::cout << std::left << std::setw(23) << row.phase << " FAILED: " << row.error
                      << '\n';
            continue;
        }
        std::cout << std::left << std::setw(23) << row.phase << std::right
                  << std::setw(6) << row.config.generation_threads
                  << std::setw(6) << row.config.prompt_threads
                  << std::setw(9) << row.result.actual_micro_batch
                  << std::setw(8) << row.result.prompt_tokens
                  << std::setw(12) << std::fixed << std::setprecision(2)
                  << row.result.prompt_tokens_per_second()
                  << std::setw(12) << row.result.generation_tokens_per_second()
                  << std::setw(13) << autotune::bytes_to_mib(row.result.memory.total_bytes());
        if (row.tradeoff_score >= 0.0) {
            std::cout << std::setw(9) << row.tradeoff_score;
        } else {
            std::cout << std::setw(9) << "-";
        }
        std::cout << '\n';
    }
}

void print_sweep_report(const autotune::SweepReport & report) {
    std::cout << "\nLocal LLM Hardware Auto-Tuner\n";
    print_model_info(report.model);
    std::cout << "Available RAM: " << autotune::bytes_to_mib(report.system_memory.available_bytes)
              << " MiB; swap used: " << std::setprecision(1)
              << 100.0 * report.system_memory.swap_used_fraction() << "%\n";
    for (const std::string & warning : report.warnings) {
        std::cout << "WARNING: " << warning << '\n';
    }

    print_sweep_table(report);
    std::cout << "\nRecommendations for this model on this machine\n";
    for (const autotune::Recommendation & recommendation : report.recommendations) {
        std::cout << "- " << recommendation.name << ": TG "
                  << recommendation.config.generation_threads << " / PP "
                  << recommendation.config.prompt_threads << " / micro-batch "
                  << recommendation.config.micro_batch;
        if (recommendation.name == "Balanced preset") {
            std::cout << " (score " << std::fixed << std::setprecision(1)
                      << recommendation.score << ")";
        }
        std::cout << "\n  " << recommendation.reason << '\n';
    }
    std::cout << "\nSelected default: generation threads "
              << report.selected_config.generation_threads << ", prompt threads "
              << report.selected_config.prompt_threads << ", prompt micro-batch "
              << report.selected_config.micro_batch << "\n";
    std::cout << "Note: allocated buffers are llama.cpp allocations, not peak process RAM.\n\n";
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const CliOptions options = parse_arguments(argc, argv);
#ifdef AUTOTUNE_GUI_AVAILABLE
        if (options.mode == Mode::Gui) {
            return autotune::run_gui(
                options.model_path,
                options.gui_auto_run,
                options.gui_smoke,
                options.gui_exit_after_run);
        }
#endif
        autotune::BackendRuntime backend(options.verbose);
        autotune::Engine engine(options.model_path);

        if (options.mode == Mode::Single) {
            const autotune::RunResult result =
                engine.run_text(options.prompt, options.single_config, true);
            print_single_result(engine.model_info(), options.single_config, result);
            return 0;
        }

        const auto progress = [](const autotune::ProgressUpdate & update) {
            if (update.has_completed_row) {
                return;
            }
            if (update.phase == "Complete") {
                std::cout << "\r[" << update.total_steps << '/' << update.total_steps
                          << "] Complete" << std::string(50, ' ') << '\n';
                return;
            }
            std::cout << "\r[" << update.completed_steps << '/' << update.total_steps << "] "
                      << update.phase << " - TG " << update.config.generation_threads << ", PP "
                      << update.config.prompt_threads << ", uBatch "
                      << update.config.micro_batch;
            if (update.repetitions > 1) {
                std::cout << " - repetition " << update.repetition << '/'
                          << update.repetitions;
            }
            std::cout << std::string(20, ' ') << std::flush;
        };

        std::atomic_bool cancel{false};
        std::atomic_bool timer_finished{false};
        std::mutex timer_mutex;
        std::condition_variable timer_condition;
        std::thread cancel_timer;
        if (options.cancel_after_ms > 0) {
            cancel_timer = std::thread([&]() {
                std::unique_lock<std::mutex> lock(timer_mutex);
                const bool benchmark_finished = timer_condition.wait_for(
                    lock,
                    std::chrono::milliseconds(options.cancel_after_ms),
                    [&timer_finished]() { return timer_finished.load(); });
                if (!benchmark_finished) {
                    cancel.store(true);
                }
            });
        }

        const auto stop_cancel_timer = [&]() {
            timer_finished.store(true);
            timer_condition.notify_all();
            if (cancel_timer.joinable()) {
                cancel_timer.join();
            }
        };

        autotune::SweepReport report;
        try {
            report = autotune::run_sweep(
                engine,
                options.sweep,
                progress,
                options.cancel_after_ms > 0 ? &cancel : nullptr);
        } catch (...) {
            stop_cancel_timer();
            throw;
        }
        stop_cancel_timer();
        if (report.cancelled) {
            std::cerr << "\nBenchmark cancelled.\n";
            return 2;
        }
        print_sweep_report(report);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
