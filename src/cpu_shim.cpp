// src/cpu_shim.cpp
#include "ggml-backend.h"
#include "ggml-cpu.h"   // declares ggml_backend_init_cpu()

extern "C" ggml_backend_t llamar_cpu_init_shim(void) {
    return ggml_backend_init_cpu();
}
