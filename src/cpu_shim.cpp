#include "ggml-backend.h"

// Match the CPU initializer's C linkage (as exported in ggml-cpu.cpp)
extern "C" ggml_backend_t ggml_backend_cpu_init(void);

extern "C" ggml_backend_t llamar_cpu_init_shim(void) {
    return ggml_backend_cpu_init();
}
