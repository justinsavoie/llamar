// cpu_shim.cpp — resolves the CPU backend symbol across ggml snapshots
extern "C" {
    #include "ggml-backend.h"
    // Some trees also ship a separate CPU header:
    #if __has_include("ggml-cpu.h")
    #  include "ggml-cpu.h"
    #endif

    // Declare both possible exports (only one exists in your tree)
    ggml_backend_t ggml_backend_cpu_init(void);   // older ggml
    ggml_backend_t ggml_backend_init_cpu(void);   // newer ggml
}

// Prefer the newer name if the header provided a prototype; otherwise call the old one.
extern "C" ggml_backend_t llamar_cpu_init_shim(void) {
    #if defined(__cplusplus) && defined(ggml_backend_init_cpu)
        return ggml_backend_init_cpu();
    #else
        // Falls back to the legacy name (present in many ggml trees)
        return ggml_backend_cpu_init();
    #endif
}
