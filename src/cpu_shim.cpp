// ---------- src/cpu_shim.cpp ----------
// Minimal portable shim for modern ggml (C++ backend only)

#include "ggml-backend.h"

// The modern C++ backend no longer provides ggml_backend_cpu_init().
// We just expose a simple entry point that returns the CPU buffer type.
// This is sufficient for any code that only needed to reference the CPU backend.

extern "C" void *llamar_cpu_buffer_type(void) {
    return (void *) ggml_backend_cpu_buffer_type();
}

// If other parts of your Rcpp glue expect a "backend initializer", you can
// export a dummy symbol to satisfy the linker but make it safe:
extern "C" ggml_backend_t llamar_cpu_init_shim(void) {
    // In the new C++ backend, there’s no need to initialize a backend struct.
    // Just return nullptr or use the buffer type directly.
    return nullptr;
}
