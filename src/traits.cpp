// Minimal stubs to avoid referencing optional extra buffer types
#include "traits.h"

namespace ggml::cpu {
tensor_traits::~tensor_traits() {}
extra_buffer_type::~extra_buffer_type() {}
} // namespace ggml::cpu

bool ggml_cpu_extra_compute_forward(struct ggml_compute_params * /*params*/, struct ggml_tensor * /*op*/) {
    return false;
}

bool ggml_cpu_extra_work_size(int /*n_threads*/, const struct ggml_tensor * /*op*/, size_t * /*size*/) {
    return false;
}
