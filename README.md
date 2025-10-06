llamar: Minimal R bindings for llama.cpp (CPU-only)

Overview

- Purpose: Provide a small, self-contained R package that loads GGUF models via llama.cpp and performs minimal, CPU-only text generation from R.
- Scope: Currently exposes a single greedy-generation function designed for simple experiments and reproducibility. Not a full-featured client.
- Upstream: Bundles a local copy of llama.cpp/ggml for consistent builds and ABI stability on all machines.

Key Features

- CPU-only inference: No external system libraries required; builds locally with R’s toolchain.
- GGUF support: Loads models in the modern GGUF format (e.g., LLaMA, Mistral, Gemma, Qwen variants converted to GGUF).
- Minimal API: One-call greedy generation for quick testing.
- Sampling controls: Temperature, top-p/k, repetition penalty (via `llama_generate`).
- Chat helpers: `chat_format()` to apply the model’s chat template and `chat()` convenience wrapper.
- Portable builds: Avoids hard-coded CPU flags that can crash on older Intel macs.

Non-goals (for now)

- No streaming token callbacks into R.
- No GPU backends (CUDA/Metal/OpenCL) enabled by default.
- No tool/function-calling abstractions beyond basic chat templating.

Requirements

- OS: macOS or Linux with a working C/C++ toolchain. Windows may work via RTools/MINGW with adjustments but is not tested here.
- R: R 4.1+ recommended. Rcpp installed (handled automatically via LinkingTo).
- Model: A local GGUF file (quantized or FP16). Example: Mistral 7B Instruct q4_k_m in GGUF.

macOS toolchain

- Ensure Xcode Command Line Tools are installed: `xcode-select --install`
- If you previously installed and hit a compile/link error, try reinstalling with a clean build: `R CMD INSTALL --preclean --clean llamar`

Installation

- From the repository root, install the package:

  - R CMD INSTALL
    - In a shell: `R CMD INSTALL llamar`

  - or from R with devtools
    - `devtools::install("llamar")`

- Verify build:
  - In R: `llamar::llama_build_test()`
  - Expected: `"Success! R package can see llama.cpp headers."`

Quick Start

1) Pick a model (GGUF) on disk. Example path:
   `/path/to/models/mistral-7b-instruct-q4_k_m.gguf`

2) Generate a short continuation:

```r
library(llamar)

model <- "/absolute/path/to/mistral-7b-instruct-q4_k_m.gguf"
prompt <- "Hello, my name is"

## Option A: greedy
txt <- llama_generate_greedy(model, prompt, n_predict = 16L, n_ctx = 256L)
cat(txt)
```

3) Sampling example

```r
# temperature / nucleus / top-k, repetition penalty, seed, stops
txt <- llama_generate(
  model, prompt,
  n_predict = 64L, n_ctx = 512L,
  temperature = 0.7, top_p = 0.9, top_k = 40L,
  repeat_penalty = 1.1, repeat_last_n = 64L,
  seed = 123, stop = c("\n\n"))
cat(txt)
```

4) Chat formatting + generation

```r
messages <- list(
  list(role = "system", content = "You are a helpful assistant."),
  list(role = "user",   content = "Explain k-means in 2 sentences.")
)
prompt <- chat_format(model, messages, add_assistant = TRUE)
txt <- llama_generate(model, prompt, n_predict = 64L, temperature = 0.7)
cat(txt)
```

Notes
- Use absolute paths without newlines or trailing spaces.
- Smaller `n_ctx` reduces memory footprint; larger values increase memory.
- `llama_generate_greedy()` performs greedy decoding (argmax).
- `llama_generate()` enables sampling (temperature/top-p/top-k), repetition penalties, seed, and stop sequences.

API Reference

- `llama_build_test()`
  - Description: Confirms that the R package is correctly linked and callable.
  - Returns: A short status string.

- `llama_generate_greedy(model, prompt, n_predict = 64L, n_ctx = 512L)`
  - `model` (character, length 1): Absolute path to a GGUF file.
  - `prompt` (character, length 1): Input prompt.
  - `n_predict` (integer): Number of tokens to generate (greedy).
  - `n_ctx` (integer): Context length (KV cache). Lower values reduce memory usage.
  - Returns: Generated continuation as a character scalar.

- `llama_generate(model, prompt, n_predict = 64L, n_ctx = 512L, temperature = 0.8, top_p = 0.95, top_k = 40L, repeat_penalty = 1.0, repeat_last_n = 64L, seed = 0L, stop = character())`
  - Adds sampling controls to the basic generator; returns text.

- `chat_format(model, messages, template = NULL, add_assistant = TRUE)`
  - Formats role-tagged messages into a single prompt using the model’s chat template. Returns a prompt string.

- `chat(model, messages, ...)`
  - Convenience: `chat_format()` + `llama_generate()` in one call.

Model Preparation

- GGUF models are required. HF transformers checkpoints must be converted to GGUF using upstream tools (outside of this package).
- Example model families often used with llama.cpp: LLaMA, Mistral, Gemma, Qwen, etc., provided as GGUF.
- Quantized models (e.g., q4_k_m) substantially reduce memory and disk size and are recommended for CPU use.

Performance & Memory

- Threads: The package auto-sets threads to the number of available hardware cores.
- Memory: `n_ctx` controls the KV cache and scales memory usage. If the OS kills R or it exits abruptly, lower `n_ctx` (e.g., 256 or 128) or use a smaller quant/model.
- Disk I/O: Models are memory-mapped where possible for faster startup.
 - Sampling cost: Adding samplers (top-p/k, penalties) introduces small overhead vs greedy; typically negligible relative to decode time on CPU.

Environment variables

- `LLAMAR_USE_MMAP`: Set to `0` to disable file memory-mapping (useful on macOS if you observe instability). Example in R:
  - `Sys.setenv(LLAMAR_USE_MMAP = "0")`
- `LLAMAR_N_THREADS`: Override the number of CPU threads used for decoding. Example:
  - `Sys.setenv(LLAMAR_N_THREADS = "4")`

Path handling

- Prefer absolute paths or expand `~` in R before passing the model path:
  - `model <- path.expand("~/path/to/model.gguf")`

Troubleshooting

- R session terminates instantly on Intel Mac
  - Cause: Illegal instruction from aggressive CPU flags (AVX2) on older CPUs.
  - Status: The build avoids forcing AVX2 by default. Reinstall the package and retry.

- “Failed to load model”
  - Check: Path is correct, absolute, and points to an existing `.gguf` file.
  - Ensure no newline or trailing spaces in the path string.

- “package or namespace load failed … shared object ‘llamar.so’ not found”
  - Fix: Update to the latest package version and reinstall with a clean build:
    - `R CMD INSTALL --preclean --clean llamar`
  - Ensure Command Line Tools are installed on macOS (see above).

- Runs out of memory or process is killed
  - Reduce `n_ctx` (e.g., 256L or 128L).
  - Use a smaller model or a more aggressive quant (e.g., q4 variants).

- macOS-specific stability notes
  - If you see unexpected termination during load/eval, try disabling mmap:
    - `Sys.setenv(LLAMAR_USE_MMAP = "0")`
  - Ensure you pass an expanded absolute path (see Path handling above).

- Output is empty or very short
  - Increase `n_predict`.
  - Your prompt may include an instruction format that ends quickly with greedy decoding. Try a different prompt.

Security and Stability Notes

- This package executes native code and memory-maps large files. Use trusted models.
- Unexpected termination usually indicates a system-level issue (e.g., OOM-kill, illegal instruction). See “Troubleshooting.”

Roadmap / Next Steps

- Streaming: incremental token callbacks surfaced into R.
- Embeddings: expose embedding extraction from llama.cpp.
- GPU support (optional builds): expose Metal (macOS) / CUDA as configurable variants.
- Logging & diagnostics: surface llama.cpp logs to R for better error visibility.

Credits

- Built on top of ggml/llama.cpp (https://github.com/ggerganov/llama.cpp) and its GGUF format.
- R interface uses Rcpp for bridging native code and R.

License

- See licenses included in the repository and upstream projects. Use is subject to the model licenses as well as ggml/llama.cpp licensing terms.
