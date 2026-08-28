#include "gui.hpp"

#include "autotuner.hpp"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <GL/gl.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace autotune {
namespace {

constexpr ImVec4 kAccent = ImVec4(0.24F, 0.78F, 0.66F, 1.0F);
constexpr ImVec4 kBlue = ImVec4(0.32F, 0.62F, 0.96F, 1.0F);
constexpr ImVec4 kGold = ImVec4(0.96F, 0.69F, 0.27F, 1.0F);
constexpr ImVec4 kRed = ImVec4(0.95F, 0.35F, 0.39F, 1.0F);
constexpr ImVec4 kMuted = ImVec4(0.57F, 0.62F, 0.70F, 1.0F);

struct WorkerSnapshot {
    bool running = false;
    bool finished = false;
    bool model_ready = false;
    bool report_ready = false;
    std::string status = "Ready";
    std::string error;
    ProgressUpdate progress;
    ModelInfo model;
    std::vector<BenchmarkRow> live_rows;
    SweepReport report;
};

class DashboardWorker {
public:
    DashboardWorker() = default;

    ~DashboardWorker() {
        request_cancel();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    DashboardWorker(const DashboardWorker &) = delete;
    DashboardWorker & operator=(const DashboardWorker &) = delete;

    void start(const std::filesystem::path & model_path, const int profile) {
        join_if_finished();
        if (worker_.joinable()) {
            return;
        }

        cancel_.store(false);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            state_ = WorkerSnapshot{};
            state_.running = true;
            state_.status = "Loading GGUF model";
        }

        SweepOptions options;
        if (profile == 1) {
            options.repetitions = 3;
            options.finalist_repetitions = 5;
        } else if (profile == 2) {
            options.smoke = true;
            options.long_prompt_tokens = 16;
            options.generation_tokens = 2;
            options.repetitions = 1;
            options.finalist_repetitions = 1;
        }

        worker_ = std::thread([this, model_path, options]() {
            try {
                BackendRuntime runtime;
                Engine engine(model_path);
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    state_.model = engine.model_info();
                    state_.model_ready = true;
                    state_.status = "Benchmarking";
                }

                const auto on_progress = [this](const ProgressUpdate & update) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    state_.progress = update;
                    state_.status = update.message;
                    if (update.has_completed_row) {
                        state_.live_rows.push_back(update.completed_row);
                    }
                };

                SweepReport report = run_sweep(engine, options, on_progress, &cancel_);
                std::lock_guard<std::mutex> lock(mutex_);
                state_.report = std::move(report);
                state_.report_ready = true;
                state_.running = false;
                state_.finished = true;
                state_.status = state_.report.cancelled ? "Cancelled" : "Complete";
            } catch (const std::exception & error) {
                std::lock_guard<std::mutex> lock(mutex_);
                state_.running = false;
                state_.finished = true;
                state_.status = "Failed";
                state_.error = error.what();
            }
        });
    }

    void request_cancel() {
        cancel_.store(true);
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_.running) {
            state_.status = "Cancelling";
        }
    }

    void join_if_finished() {
        bool should_join = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            should_join = state_.finished && !state_.running;
        }
        if (should_join && worker_.joinable()) {
            worker_.join();
        }
    }

    [[nodiscard]] WorkerSnapshot snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return state_;
    }

private:
    mutable std::mutex mutex_;
    WorkerSnapshot state_;
    std::atomic_bool cancel_{false};
    std::thread worker_;
};

struct ChartBar {
    int order = 0;
    std::string label;
    std::string detail;
    double value = 0.0;
};

std::string format_decimal(const double value, const int precision = 1) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(precision) << value;
    return output.str();
}

std::string format_gib(const std::uint64_t bytes) {
    return format_decimal(static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0), 2) + " GiB";
}

void apply_dashboard_style(const float scale) {
    ImGui::StyleColorsDark();
    ImGuiStyle & style = ImGui::GetStyle();
    style.WindowRounding = 0.0F;
    style.ChildRounding = 10.0F;
    style.FrameRounding = 7.0F;
    style.GrabRounding = 7.0F;
    style.TabRounding = 7.0F;
    style.WindowPadding = ImVec2(18.0F, 16.0F);
    style.FramePadding = ImVec2(10.0F, 7.0F);
    style.ItemSpacing = ImVec2(10.0F, 9.0F);
    style.ScrollbarSize = 13.0F;
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.045F, 0.055F, 0.075F, 1.0F);
    style.Colors[ImGuiCol_ChildBg] = ImVec4(0.070F, 0.082F, 0.105F, 1.0F);
    style.Colors[ImGuiCol_Border] = ImVec4(0.17F, 0.20F, 0.25F, 1.0F);
    style.Colors[ImGuiCol_FrameBg] = ImVec4(0.095F, 0.11F, 0.14F, 1.0F);
    style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.13F, 0.16F, 0.20F, 1.0F);
    style.Colors[ImGuiCol_Button] = ImVec4(0.16F, 0.52F, 0.45F, 1.0F);
    style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.20F, 0.64F, 0.55F, 1.0F);
    style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.13F, 0.43F, 0.37F, 1.0F);
    style.Colors[ImGuiCol_Header] = ImVec4(0.13F, 0.34F, 0.31F, 1.0F);
    style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.17F, 0.45F, 0.40F, 1.0F);
    style.Colors[ImGuiCol_PlotHistogram] = kAccent;
    style.ScaleAllSizes(scale);
    style.FontScaleDpi = scale;
}

void draw_status_badge(const WorkerSnapshot & snapshot) {
    ImVec4 color = kMuted;
    if (snapshot.running) {
        color = kBlue;
    } else if (!snapshot.error.empty()) {
        color = kRed;
    } else if (snapshot.report_ready && !snapshot.report.cancelled) {
        color = kAccent;
    } else if (snapshot.report_ready && snapshot.report.cancelled) {
        color = kGold;
    }
    ImGui::TextColored(color, "[ %s ]", snapshot.status.c_str());
}

void draw_warning(const std::string & text, const ImVec4 & color) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(color.x * 0.12F, color.y * 0.12F,
                                                   color.z * 0.12F, 1.0F));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(color.x, color.y, color.z, 0.48F));
    ImGui::BeginChild(
        ("warning-" + text).c_str(), ImVec2(0.0F, 48.0F), ImGuiChildFlags_Borders);
    ImGui::TextColored(color, "!  %s", text.c_str());
    ImGui::EndChild();
    ImGui::PopStyleColor(2);
}

void draw_bar_chart(
    const char * id,
    const char * title,
    std::vector<ChartBar> bars,
    const ImVec4 & color) {
    std::sort(bars.begin(), bars.end(), [](const ChartBar & left, const ChartBar & right) {
        return left.order < right.order;
    });

    ImGui::BeginChild(id, ImVec2(0.0F, 225.0F), ImGuiChildFlags_Borders);
    ImGui::TextUnformatted(title);
    ImGui::Separator();
    if (bars.empty()) {
        ImGui::TextColored(kMuted, "Results appear here as candidates finish.");
        ImGui::EndChild();
        return;
    }

    const double maximum = std::max_element(
        bars.begin(), bars.end(), [](const ChartBar & left, const ChartBar & right) {
            return left.value < right.value;
        })->value;
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const ImVec2 available = ImGui::GetContentRegionAvail();
    const float footer = 27.0F;
    const float top_padding = 22.0F;
    const float graph_height = std::max(40.0F, available.y - footer - top_padding);
    const float gap = 8.0F;
    const float bar_width =
        std::max(14.0F, (available.x - gap * static_cast<float>(bars.size() + 1)) /
                            static_cast<float>(bars.size()));
    ImDrawList * draw = ImGui::GetWindowDrawList();
    draw->AddLine(
        ImVec2(origin.x, origin.y + top_padding + graph_height),
        ImVec2(origin.x + available.x, origin.y + top_padding + graph_height),
        ImGui::GetColorU32(ImVec4(0.28F, 0.31F, 0.37F, 1.0F)),
        1.0F);

    for (std::size_t index = 0; index < bars.size(); ++index) {
        const ChartBar & bar = bars[index];
        const float ratio = maximum > 0.0 ? static_cast<float>(bar.value / maximum) : 0.0F;
        const float x0 = origin.x + gap + static_cast<float>(index) * (bar_width + gap);
        const float y1 = origin.y + top_padding + graph_height;
        const float y0 = y1 - graph_height * ratio;
        const ImVec2 minimum(x0, y0);
        const ImVec2 maximum_point(x0 + bar_width, y1);
        draw->AddRectFilled(minimum, maximum_point, ImGui::GetColorU32(color), 4.0F);

        const std::string value = format_decimal(bar.value, 1);
        const ImVec2 value_size = ImGui::CalcTextSize(value.c_str());
        draw->AddText(
            ImVec2(x0 + (bar_width - value_size.x) * 0.5F, std::max(origin.y, y0 - 18.0F)),
            ImGui::GetColorU32(ImGuiCol_Text),
            value.c_str());
        const ImVec2 label_size = ImGui::CalcTextSize(bar.label.c_str());
        draw->AddText(
            ImVec2(x0 + (bar_width - label_size.x) * 0.5F, y1 + 6.0F),
            ImGui::GetColorU32(kMuted),
            bar.label.c_str());

        if (ImGui::IsMouseHoveringRect(minimum, maximum_point)) {
            ImGui::BeginTooltip();
            ImGui::Text("%s: %.2f", bar.label.c_str(), bar.value);
            ImGui::TextUnformatted(bar.detail.c_str());
            ImGui::EndTooltip();
        }
    }
    ImGui::Dummy(available);
    ImGui::EndChild();
}

std::vector<ChartBar> generation_bars(const std::vector<BenchmarkRow> & rows) {
    std::vector<ChartBar> bars;
    for (const BenchmarkRow & row : rows) {
        if (row.succeeded && row.phase == "Generation threads") {
            bars.push_back(ChartBar{
                row.config.generation_threads,
                std::to_string(row.config.generation_threads) + "T",
                "Generation tokens per second",
                row.result.generation_tokens_per_second(),
            });
        }
    }
    return bars;
}

std::vector<ChartBar> micro_batch_bars(const std::vector<BenchmarkRow> & rows) {
    std::vector<ChartBar> bars;
    for (const BenchmarkRow & row : rows) {
        if (row.succeeded && row.phase == "Prompt micro-batch") {
            bars.push_back(ChartBar{
                row.result.actual_micro_batch,
                "u" + std::to_string(row.result.actual_micro_batch),
                "Prompt tokens per second",
                row.result.prompt_tokens_per_second(),
            });
        }
    }
    return bars;
}

std::vector<ChartBar> tradeoff_bars(const std::vector<BenchmarkRow> & rows) {
    std::vector<ChartBar> bars;
    int order = 0;
    for (const BenchmarkRow & row : rows) {
        if (row.succeeded && row.phase.rfind("Finalist short - ", 0) == 0 &&
            row.tradeoff_score >= 0.0) {
            const std::string preset = row.phase.substr(std::string("Finalist short - ").size());
            bars.push_back(ChartBar{
                order++,
                "T" + std::to_string(row.config.generation_threads) + "/P" +
                    std::to_string(row.config.prompt_threads) + "/u" +
                    std::to_string(row.result.actual_micro_batch),
                preset + " balanced score",
                row.tradeoff_score,
            });
        }
    }
    return bars;
}

void draw_results_table(const std::vector<BenchmarkRow> & rows) {
    if (rows.empty()) {
        ImGui::TextColored(kMuted, "Measured rows will stream into this table.");
        return;
    }

    constexpr ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                      ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY |
                                      ImGuiTableFlags_SizingStretchProp;
    if (ImGui::BeginTable("results-table", 9, flags, ImVec2(0.0F, 285.0F))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Stage", ImGuiTableColumnFlags_WidthStretch, 2.3F);
        ImGui::TableSetupColumn("TG", ImGuiTableColumnFlags_WidthFixed, 44.0F);
        ImGui::TableSetupColumn("PP", ImGuiTableColumnFlags_WidthFixed, 44.0F);
        ImGui::TableSetupColumn("uBatch", ImGuiTableColumnFlags_WidthFixed, 62.0F);
        ImGui::TableSetupColumn("Prompt", ImGuiTableColumnFlags_WidthFixed, 62.0F);
        ImGui::TableSetupColumn("PP tok/s", ImGuiTableColumnFlags_WidthFixed, 78.0F);
        ImGui::TableSetupColumn("TG tok/s", ImGuiTableColumnFlags_WidthFixed, 78.0F);
        ImGui::TableSetupColumn("Buffers", ImGuiTableColumnFlags_WidthFixed, 78.0F);
        ImGui::TableSetupColumn("Score", ImGuiTableColumnFlags_WidthFixed, 62.0F);
        ImGui::TableHeadersRow();

        for (const BenchmarkRow & row : rows) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            if (!row.succeeded) {
                ImGui::TextColored(kRed, "%s", row.phase.c_str());
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("%s", row.error.c_str());
                }
                ImGui::TableSetColumnIndex(1);
                ImGui::TextColored(kRed, "FAILED");
                continue;
            }
            ImGui::TextUnformatted(row.phase.c_str());
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%d", row.config.generation_threads);
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%d", row.config.prompt_threads);
            ImGui::TableSetColumnIndex(3);
            ImGui::Text("%d", row.result.actual_micro_batch);
            ImGui::TableSetColumnIndex(4);
            ImGui::Text("%d", row.result.prompt_tokens);
            ImGui::TableSetColumnIndex(5);
            ImGui::Text("%.1f", row.result.prompt_tokens_per_second());
            ImGui::TableSetColumnIndex(6);
            ImGui::Text("%.1f", row.result.generation_tokens_per_second());
            ImGui::TableSetColumnIndex(7);
            ImGui::Text("%.0f MB", bytes_to_mib(row.result.memory.total_bytes()));
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("llama.cpp allocated buffers, not peak process RAM");
            }
            ImGui::TableSetColumnIndex(8);
            if (row.tradeoff_score >= 0.0) {
                ImGui::TextColored(kAccent, "%.1f", row.tradeoff_score);
            } else {
                ImGui::TextColored(kMuted, "-");
            }
        }
        ImGui::EndTable();
    }
}

void draw_recommendations(const SweepReport & report) {
    if (report.recommendations.empty()) {
        return;
    }
    ImGui::SeparatorText("Recommended presets");
    const int columns = ImGui::GetContentRegionAvail().x > 1000.0F ? 4 : 2;
    if (ImGui::BeginTable("recommendation-cards", columns, ImGuiTableFlags_SizingStretchSame)) {
        for (std::size_t index = 0; index < report.recommendations.size(); ++index) {
            const Recommendation & recommendation = report.recommendations[index];
            ImGui::TableNextColumn();
            const bool balanced = recommendation.name == "Balanced preset";
            if (balanced) {
                ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.07F, 0.18F, 0.16F, 1.0F));
                ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(kAccent.x, kAccent.y, kAccent.z, 0.7F));
            }
            const std::string child_id = "recommendation-" + std::to_string(index);
            ImGui::BeginChild(child_id.c_str(), ImVec2(0.0F, 146.0F), ImGuiChildFlags_Borders);
            ImGui::TextColored(balanced ? kAccent : kBlue, "%s", recommendation.name.c_str());
            ImGui::Text("TG %d   PP %d   uBatch %d",
                        recommendation.config.generation_threads,
                        recommendation.config.prompt_threads,
                        recommendation.config.micro_batch);
            if (balanced) {
                ImGui::Text("Score %.1f / 100", recommendation.score);
            }
            ImGui::Spacing();
            ImGui::TextWrapped("%s", recommendation.reason.c_str());
            ImGui::EndChild();
            if (balanced) {
                ImGui::PopStyleColor(2);
            }
        }
        ImGui::EndTable();
    }
}

void render_dashboard(
    DashboardWorker & worker,
    std::array<char, 2048> & model_path,
    int & profile) {
    worker.join_if_finished();
    const WorkerSnapshot snapshot = worker.snapshot();
    const std::vector<BenchmarkRow> & rows =
        snapshot.report_ready ? snapshot.report.rows : snapshot.live_rows;

    ImGuiIO & io = ImGui::GetIO();
    ImGui::SetNextWindowPos(ImVec2(0.0F, 0.0F));
    ImGui::SetNextWindowSize(io.DisplaySize);
    constexpr ImGuiWindowFlags window_flags =
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus;
    ImGui::Begin("Local LLM Auto-Tuner", nullptr, window_flags);

    ImGui::TextColored(kAccent, "LOCAL LLM HARDWARE AUTO-TUNER");
    ImGui::SameLine(ImGui::GetWindowWidth() - 180.0F);
    draw_status_badge(snapshot);
    ImGui::SetWindowFontScale(1.55F);
    ImGui::TextUnformatted("Find the fastest llama.cpp settings for this machine");
    ImGui::SetWindowFontScale(1.0F);
    ImGui::TextColored(
        kMuted,
        "One GGUF, fixed workloads, real CPU inference. No network or cloud services.");
    ImGui::Spacing();

    if (ImGui::BeginTable("top-panels", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Model", ImGuiTableColumnFlags_WidthStretch, 2.0F);
        ImGui::TableSetupColumn("Run", ImGuiTableColumnFlags_WidthStretch, 1.0F);
        ImGui::TableNextColumn();
        ImGui::BeginChild("model-panel", ImVec2(0.0F, 184.0F), ImGuiChildFlags_Borders);
        ImGui::TextColored(kBlue, "MODEL");
        ImGui::BeginDisabled(snapshot.running);
        ImGui::SetNextItemWidth(-FLT_MIN);
        ImGui::InputText("##model-path", model_path.data(), model_path.size());
        ImGui::EndDisabled();
        if (snapshot.model_ready) {
            ImGui::Text("%s", snapshot.model.description.c_str());
            ImGui::TextColored(
                kMuted,
                "%.2fB parameters  |  %s  |  loaded in %.0f ms",
                static_cast<double>(snapshot.model.parameter_count) / 1'000'000'000.0,
                format_gib(snapshot.model.file_bytes).c_str(),
                snapshot.model.load_time_ms);
        } else {
            std::error_code error;
            const auto bytes = std::filesystem::file_size(model_path.data(), error);
            if (error) {
                ImGui::TextColored(kRed, "Model file not found or unreadable.");
            } else {
                ImGui::TextColored(kMuted, "%s on disk", format_gib(bytes).c_str());
            }
        }
        ImGui::Spacing();
        ImGui::TextWrapped(
            "CPU-only tuning keeps GPU state out of the comparison. The model stays fixed; "
            "only generation threads, prompt threads, and prompt micro-batch change.");
        ImGui::EndChild();

        ImGui::TableNextColumn();
        ImGui::BeginChild("run-panel", ImVec2(0.0F, 184.0F), ImGuiChildFlags_Borders);
        ImGui::TextColored(kBlue, "BENCHMARK PROFILE");
        ImGui::BeginDisabled(snapshot.running);
        ImGui::SetNextItemWidth(-FLT_MIN);
        const char * profiles[] = {
            "Quick - 1 exploratory / 3 finalist reps",
            "Thorough - 3 exploratory / 5 finalist reps",
            "Smoke test - tiny workloads",
        };
        ImGui::Combo("##profile", &profile, profiles, 3);
        ImGui::EndDisabled();
        ImGui::Spacing();
        if (!snapshot.running) {
            if (ImGui::Button("Run Auto-Tune", ImVec2(-FLT_MIN, 42.0F))) {
                worker.start(model_path.data(), profile);
            }
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.52F, 0.16F, 0.18F, 1.0F));
            if (ImGui::Button("Cancel", ImVec2(-FLT_MIN, 42.0F))) {
                worker.request_cancel();
            }
            ImGui::PopStyleColor();
        }
        ImGui::TextColored(
            kMuted,
            profile == 1 ? "Most stable; expect several minutes."
                         : profile == 2 ? "Verifies the pipeline in seconds."
                                        : "Fast search with stable finalist medians.");
        ImGui::EndChild();
        ImGui::EndTable();
    }

    if (!snapshot.error.empty()) {
        draw_warning(snapshot.error, kRed);
    }
    if (snapshot.report_ready) {
        for (const std::string & warning : snapshot.report.warnings) {
            draw_warning(warning, kGold);
        }
    } else {
        const SystemMemoryInfo memory = read_system_memory();
        if (memory.swap_used_fraction() > 0.8) {
            draw_warning("Swap is over 80% used; close memory-heavy apps for steadier results.", kGold);
        }
    }

    ImGui::SeparatorText("Live benchmark");
    if (snapshot.running || snapshot.progress.total_steps > 0) {
        const float fraction = snapshot.progress.total_steps > 0
            ? static_cast<float>(snapshot.progress.completed_steps) /
                  static_cast<float>(snapshot.progress.total_steps)
            : 0.0F;
        const std::string overlay = std::to_string(snapshot.progress.completed_steps) + " / " +
                                    std::to_string(snapshot.progress.total_steps) + " steps";
        ImGui::ProgressBar(std::clamp(fraction, 0.0F, 1.0F), ImVec2(-FLT_MIN, 0.0F), overlay.c_str());
        if (!snapshot.progress.phase.empty() && snapshot.progress.phase != "Complete") {
            ImGui::Text(
                "%s  |  generation threads %d  |  prompt threads %d  |  micro-batch %d",
                snapshot.progress.phase.c_str(),
                snapshot.progress.config.generation_threads,
                snapshot.progress.config.prompt_threads,
                snapshot.progress.config.micro_batch);
        }
    } else {
        ImGui::TextColored(kMuted, "Choose a profile and click Run Auto-Tune.");
    }

    const auto draw_charts = [&rows]() {
        ImGui::SeparatorText("Tradeoff dashboard");
        const int chart_columns = ImGui::GetContentRegionAvail().x > 1080.0F ? 3 : 1;
        if (ImGui::BeginTable("charts", chart_columns, ImGuiTableFlags_SizingStretchSame)) {
            ImGui::TableNextColumn();
            draw_bar_chart(
                "generation-chart", "Generation speed (tok/s)", generation_bars(rows), kBlue);
            ImGui::TableNextColumn();
            draw_bar_chart(
                "micro-chart", "Long-prompt speed (tok/s)", micro_batch_bars(rows), kAccent);
            ImGui::TableNextColumn();
            draw_bar_chart(
                "score-chart", "Validated tradeoff score", tradeoff_bars(rows), kGold);
            ImGui::EndTable();
        }
    };

    if (snapshot.report_ready && !snapshot.report.cancelled) {
        draw_recommendations(snapshot.report);
        draw_charts();
        ImGui::TextColored(
            kMuted,
            "Best means best for this GGUF, these fixed workloads, and this machine. "
            "Buffers are llama.cpp allocations, not peak process RAM.");
        ImGui::SeparatorText("All measured candidates");
        draw_results_table(rows);
    } else {
        draw_results_table(rows);
        draw_charts();
    }
    ImGui::End();
}

void glfw_error_callback(const int error, const char * description) {
    std::cerr << "GLFW error " << error << ": " << description << '\n';
}

}  // namespace

int run_gui(
    const std::filesystem::path & initial_model_path,
    const bool auto_run,
    const bool smoke_profile,
    const bool exit_after_run) {
    glfwSetErrorCallback(glfw_error_callback);
    if (glfwInit() == GLFW_FALSE) {
        std::cerr << "error: GLFW could not initialize a display\n";
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    GLFWmonitor * primary_monitor = glfwGetPrimaryMonitor();
    const float scale = primary_monitor != nullptr
        ? ImGui_ImplGlfw_GetContentScaleForMonitor(primary_monitor)
        : 1.0F;
    GLFWwindow * window = glfwCreateWindow(
        static_cast<int>(1440.0F * scale),
        static_cast<int>(900.0F * scale),
        "Local LLM Hardware Auto-Tuner",
        nullptr,
        nullptr);
    if (window == nullptr) {
        glfwTerminate();
        std::cerr << "error: GLFW could not create the OpenGL window\n";
        return 1;
    }
    glfwSetWindowSizeLimits(window, 980, 680, GLFW_DONT_CARE, GLFW_DONT_CARE);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO & io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    apply_dashboard_style(scale);
    if (!ImGui_ImplGlfw_InitForOpenGL(window, true)) {
        ImGui::DestroyContext();
        glfwDestroyWindow(window);
        glfwTerminate();
        std::cerr << "error: Dear ImGui GLFW backend could not initialize\n";
        return 1;
    }
    if (!ImGui_ImplOpenGL3_Init("#version 130")) {
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glfwDestroyWindow(window);
        glfwTerminate();
        std::cerr << "error: Dear ImGui OpenGL backend could not initialize\n";
        return 1;
    }

    int exit_code = 0;
    {
        DashboardWorker worker;
        std::array<char, 2048> model_path{};
        std::snprintf(model_path.data(), model_path.size(), "%s", initial_model_path.c_str());
        int profile = smoke_profile ? 2 : 0;
        if (auto_run) {
            worker.start(model_path.data(), profile);
        }

        while (glfwWindowShouldClose(window) == GLFW_FALSE) {
            glfwPollEvents();
            if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0) {
                ImGui_ImplGlfw_Sleep(10);
                continue;
            }

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            render_dashboard(worker, model_path, profile);

            ImGui::Render();
            int display_width = 0;
            int display_height = 0;
            glfwGetFramebufferSize(window, &display_width, &display_height);
            glViewport(0, 0, display_width, display_height);
            glClearColor(0.035F, 0.043F, 0.058F, 1.0F);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(window);

            const WorkerSnapshot snapshot = worker.snapshot();
            if (exit_after_run && auto_run && snapshot.finished) {
                if (!snapshot.error.empty()) {
                    std::cerr << "error: " << snapshot.error << '\n';
                    exit_code = 1;
                } else if (snapshot.report.cancelled) {
                    exit_code = 2;
                }
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }
        }
        worker.request_cancel();
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return exit_code;
}

}  // namespace autotune
