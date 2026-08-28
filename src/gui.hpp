#pragma once

#include <filesystem>

namespace autotune {

int run_gui(
    const std::filesystem::path & initial_model_path,
    bool auto_run = false,
    bool smoke_profile = false,
    bool exit_after_run = false);

}  // namespace autotune
