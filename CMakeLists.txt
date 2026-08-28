cmake_minimum_required(VERSION 3.20)

project(llama_autotune VERSION 0.1.0 LANGUAGES C CXX)

option(AUTOTUNE_BUILD_GUI "Build the Dear ImGui dashboard" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static dependencies" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "Do not build llama.cpp common helpers" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Do not build llama.cpp tests" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "Do not build llama.cpp tools" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Do not build llama.cpp examples" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Do not build llama.cpp server" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "Do not build llama.cpp app" FORCE)
set(GGML_NATIVE ON CACHE BOOL "Optimize ggml for this machine" FORCE)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
    message(FATAL_ERROR "Missing vendored llama.cpp dependency at third_party/llama.cpp")
endif()

add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)

add_executable(llama-autotune
    src/main.cpp
    src/autotuner.cpp
)

target_include_directories(llama-autotune PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/src"
)

target_link_libraries(llama-autotune PRIVATE llama)

find_package(Threads REQUIRED)
target_link_libraries(llama-autotune PRIVATE Threads::Threads)

if(AUTOTUNE_BUILD_GUI)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui/imgui.cpp")
        message(FATAL_ERROR "Missing vendored Dear ImGui dependency at third_party/imgui")
    endif()

    set(_autotune_gui_prefix "${CMAKE_CURRENT_SOURCE_DIR}/third_party/gui-deps/usr")
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
       CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$" AND
       EXISTS "${_autotune_gui_prefix}/lib/x86_64-linux-gnu/cmake/glfw3/glfw3Config.cmake")
        list(PREPEND CMAKE_PREFIX_PATH "${_autotune_gui_prefix}")
        set(_autotune_uses_bundled_gui ON)
    endif()

    set(OpenGL_GL_PREFERENCE GLVND)
    find_package(glfw3 3.3 CONFIG REQUIRED)
    find_package(OpenGL REQUIRED)

    add_library(autotune-imgui STATIC
        third_party/imgui/imgui.cpp
        third_party/imgui/imgui_draw.cpp
        third_party/imgui/imgui_tables.cpp
        third_party/imgui/imgui_widgets.cpp
        third_party/imgui/backends/imgui_impl_glfw.cpp
        third_party/imgui/backends/imgui_impl_opengl3.cpp
    )
    target_include_directories(autotune-imgui PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui/backends"
    )
    target_compile_definitions(autotune-imgui PUBLIC GLFW_INCLUDE_NONE)
    target_link_libraries(autotune-imgui PUBLIC glfw OpenGL::GL)

    target_sources(llama-autotune PRIVATE src/gui.cpp)
    target_compile_definitions(llama-autotune PRIVATE AUTOTUNE_GUI_AVAILABLE=1)
    target_link_libraries(llama-autotune PRIVATE autotune-imgui)

    if(_autotune_uses_bundled_gui)
        set(_autotune_gui_libdir "${_autotune_gui_prefix}/lib/x86_64-linux-gnu")
        add_custom_command(TARGET llama-autotune POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E make_directory
                "$<TARGET_FILE_DIR:llama-autotune>/gui-libs"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${_autotune_gui_libdir}/libglfw.so.3.3"
                "$<TARGET_FILE_DIR:llama-autotune>/gui-libs/libglfw.so.3"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${_autotune_gui_libdir}/libOpenGL.so.0.0.0"
                "$<TARGET_FILE_DIR:llama-autotune>/gui-libs/libOpenGL.so.0"
            COMMENT "Copying bundled GUI runtime libraries beside llama-autotune"
        )
        set_target_properties(llama-autotune PROPERTIES
            BUILD_RPATH "$ORIGIN/gui-libs"
            INSTALL_RPATH "$ORIGIN/gui-libs"
            BUILD_WITH_INSTALL_RPATH ON
        )
        install(
            FILES "${_autotune_gui_libdir}/libglfw.so.3.3"
            DESTINATION bin/gui-libs
            RENAME libglfw.so.3
        )
        install(
            FILES "${_autotune_gui_libdir}/libOpenGL.so.0.0.0"
            DESTINATION bin/gui-libs
            RENAME libOpenGL.so.0
        )
    endif()
endif()

if(MSVC)
    target_compile_options(llama-autotune PRIVATE /W4 /permissive-)
else()
    target_compile_options(llama-autotune PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS llama-autotune RUNTIME DESTINATION bin)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME cli-help COMMAND llama-autotune --help)
    add_test(
        NAME cli-rejects-invalid-batch
        COMMAND llama-autotune --single --batch 64 --micro-batch 256
    )
    set_tests_properties(cli-rejects-invalid-batch PROPERTIES WILL_FAIL TRUE)
    add_test(
        NAME cli-reports-missing-model
        COMMAND llama-autotune --single --model /definitely/missing/model.gguf
    )
    set_tests_properties(cli-reports-missing-model PROPERTIES WILL_FAIL TRUE)
endif()
