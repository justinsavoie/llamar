// cpu_shim.cpp — resolve CPU backend across ggml versions
extern "C" {
    #include "ggml-backend.h"
    // include the CPU header if present (some trees have only ggml-backend.h)
    #if __has_include("ggml-cpu.h")
    #   include "ggml-cpu.h"
    #endif

    // forward-declare both possible exports (only one will exist in your tree)
    ggml_backend_t ggml_backend_cpu_init(void);   // older ggml
    ggml_backend_t ggml_backend_init_cpu(void);   // newer ggml
}

// Use whichever one the headers actually declared.
// We can’t test for symbol presence at link time, so use
// preprocessor detection against the header content.
extern "C" ggml_backend_t llamar_cpu_init_shim(void) {
    // prefer the newer name when it’s declared
    #if defined(__cplusplus)
        // if the newer function is declared in ggml-cpu.h, there will be a prototype
        #if defined(ggml_backend_init_cpu) || (defined(__has_include) && __has_include("ggml-cpu.h"))
            // Many trees still only have the old name; call and let the linker resolve the one that exists.
            // Try new name first if it’s available to the compiler:
            return ggml_backend_init_cpu();
        #else
            return ggml_backend_cpu_init();
        #endif
    #else
        return ggml_backend_cpu_init();
    #endif
}
