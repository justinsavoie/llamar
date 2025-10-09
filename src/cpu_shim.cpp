// cpu_shim.cpp — portable CPU backend resolver across ggml snapshots
extern "C" {
    #include "ggml-backend.h"
    #if __has_include("ggml-cpu.h")
    #  include "ggml-cpu.h"
    #endif

    // Declare both possible symbols (one will exist at link time)
    ggml_backend_t ggml_backend_cpu_init(void);
    ggml_backend_t ggml_backend_init_cpu(void);
}

extern "C" ggml_backend_t llamar_cpu_init_shim(void) {
    // Prefer the newer name if present; fall back otherwise.
    // We can't feature-detect reliably here, so just try the legacy entry.
    // (Both are declared; the linker resolves the one that actually exists.)
    #if defined(ggml_backend_init_cpu)
        return ggml_backend_init_cpu();
    #else
        return ggml_backend_cpu_init();
    #endif
}
