#include <Rcpp.h>
#include "llama.h"
#include <thread>
#include <random>

using namespace Rcpp;

extern "C" SEXP llama_build_test();
extern "C" SEXP llama_generate_greedy(SEXP model_path_, SEXP prompt_, SEXP n_predict_, SEXP n_ctx_);
extern "C" SEXP llama_generate_sampled(SEXP model_path_, SEXP prompt_, SEXP n_predict_, SEXP n_ctx_,
                                        SEXP temperature_, SEXP top_p_, SEXP top_k_,
                                        SEXP repeat_penalty_, SEXP repeat_last_n_, SEXP seed_, SEXP stop_);
extern "C" SEXP llama_chat_format(SEXP model_path_, SEXP roles_, SEXP contents_, SEXP tmpl_, SEXP add_assistant_);

SEXP llama_build_test() {
  return Rf_mkString("Success! R package can see llama.cpp headers.");
}

// Minimal CPU-only greedy generation using llama.cpp C API
SEXP llama_generate_greedy(SEXP model_path_, SEXP prompt_, SEXP n_predict_, SEXP n_ctx_) {
  try {
    std::string model_path = as<std::string>(model_path_);
    std::string prompt     = as<std::string>(prompt_);
    int n_predict          = as<int>(n_predict_);
    int n_ctx              = as<int>(n_ctx_);

    if (model_path.empty()) {
      Rcpp::stop("Model path is empty");
    }

    if (n_predict <= 0) {
      return Rf_mkString("");
    }

    // Initialize backend
    llama_backend_init();

    // Load model
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU-only
    // allow disabling mmap via env var: LLAMAR_USE_MMAP=0
    {
      bool use_mmap = true;
      if (const char *e = std::getenv("LLAMAR_USE_MMAP")) {
        use_mmap = std::atoi(e) != 0;
      }
      mparams.use_mmap  = use_mmap;
    }
    mparams.use_mlock    = false;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
      Rcpp::stop(std::string("Failed to load model: ") + model_path);
    }

    // Create context
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = std::max(8, n_ctx <= 0 ? 512 : n_ctx);
    // CPU-only: ensure no device offloading paths are taken
    cparams.offload_kqv     = false;
    cparams.op_offload      = false;
    // allow overriding thread count via env var: LLAMAR_N_THREADS
    {
      unsigned t = std::thread::hardware_concurrency();
      if (const char *e = std::getenv("LLAMAR_N_THREADS")) {
        int v = std::atoi(e);
        if (v > 0) t = (unsigned)v;
      }
      cparams.n_threads = std::max(1u, t);
    }
    cparams.n_threads_batch = cparams.n_threads;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
      llama_model_free(model);
      Rcpp::stop("Failed to create llama context");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab     = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0) {
      llama_free(ctx);
      llama_model_free(model);
      Rcpp::stop("Invalid vocabulary size from model");
    }
    if (n_vocab <= 0) {
      llama_free(ctx);
      llama_model_free(model);
      Rcpp::stop("Invalid vocabulary size from model");
    }

    // Tokenize prompt
    std::vector<llama_token> tokens;
    tokens.resize(std::max<int>(32, prompt.size() + 8));
    int32_t ntok = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(), tokens.data(), (int32_t)tokens.size(), /*add_special*/ true, /*parse_special*/ false);
    if (ntok < 0) {
      tokens.resize(-ntok);
      ntok = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(), tokens.data(), (int32_t)tokens.size(), /*add_special*/ true, /*parse_special*/ false);
    }
    tokens.resize(ntok);

    // Feed prompt
    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
    int32_t rc = llama_decode(ctx, batch);
    if (rc < 0) {
      llama_free(ctx);
      llama_model_free(model);
      Rcpp::stop(std::string("llama_decode failed on prompt (rc=") + std::to_string(rc) + ")");
    }

    // Greedy generation loop
    std::string generated;
    generated.reserve(1024);
    std::vector<char> piece(4096);

    for (int i = 0; i < n_predict; ++i) {
      llama_token last = tokens.back();
      llama_batch b = llama_batch_get_one(&last, 1);
      // request logits for this token
      if (b.logits) b.logits[0] = 1;

      rc = llama_decode(ctx, b);
      if (rc < 0) {
        break;
      }

      float * logits = llama_get_logits(ctx);
      if (logits == nullptr) {
        // no logits available; abort generation gracefully
        break;
      }
      // argmax over vocabulary
      int best_id = 0;
      float best_logit = logits[0];
      for (int vid = 1; vid < n_vocab; ++vid) {
        if (logits[vid] > best_logit) {
          best_logit = logits[vid];
          best_id = vid;
        }
      }

      // append token and detokenize piece
      tokens.push_back((llama_token)best_id);

      // stop on end-of-generation tokens (EOS/EOT)
      if (llama_vocab_is_eog(vocab, (llama_token)best_id)) {
        break;
      }

      int32_t n = llama_token_to_piece(vocab, (llama_token)best_id, piece.data(), (int32_t)piece.size(), /*lstrip*/ 0, /*special*/ true);
      if (n > 0) {
        generated.append(piece.data(), piece.data() + n);
      } else if (n == (int32_t)piece.size()) {
        // enlarge buffer and retry once
        piece.resize(piece.size() * 2);
        n = llama_token_to_piece(vocab, (llama_token)best_id, piece.data(), (int32_t)piece.size(), 0, true);
        if (n > 0) {
          generated.append(piece.data(), piece.data() + n);
        }
      }
    }

    // cleanup
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return Rcpp::wrap(generated);
  } catch (std::exception &e) {
    Rcpp::stop(std::string("llama_generate_greedy error: ") + e.what());
  } catch (...) {
    Rcpp::stop("llama_generate_greedy: unknown error");
  }
}

// Sampling-enabled generation using llama.cpp sampler chain
SEXP llama_generate_sampled(SEXP model_path_, SEXP prompt_, SEXP n_predict_, SEXP n_ctx_,
                            SEXP temperature_, SEXP top_p_, SEXP top_k_,
                            SEXP repeat_penalty_, SEXP repeat_last_n_, SEXP seed_, SEXP stop_) {
  try {
    std::string model_path = as<std::string>(model_path_);
    std::string prompt     = as<std::string>(prompt_);
    int n_predict          = as<int>(n_predict_);
    int n_ctx              = as<int>(n_ctx_);
    double temperature     = as<double>(temperature_);
    double top_p           = as<double>(top_p_);
    int top_k              = as<int>(top_k_);
    double repeat_penalty  = as<double>(repeat_penalty_);
    int repeat_last_n      = as<int>(repeat_last_n_);
    int seed               = as<int>(seed_);

    std::vector<std::string> stops;
    if (!Rf_isNull(stop_)) {
      CharacterVector sv(stop_);
      for (int i = 0; i < sv.size(); ++i) {
        if (sv[i] != NA_STRING) stops.emplace_back(as<std::string>(sv[i]));
      }
    }

    if (model_path.empty()) {
      Rcpp::stop("Model path is empty");
    }

    if (n_predict <= 0) {
      return Rf_mkString("");
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    {
      bool use_mmap = true;
      if (const char *e = std::getenv("LLAMAR_USE_MMAP")) {
        use_mmap = std::atoi(e) != 0;
      }
      mparams.use_mmap  = use_mmap;
    }
    mparams.use_mlock    = false;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
      Rcpp::stop(std::string("Failed to load model: ") + model_path);
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = std::max(8, n_ctx <= 0 ? 512 : n_ctx);
    // CPU-only: ensure no device offloading paths are taken
    cparams.offload_kqv     = false;
    cparams.op_offload      = false;
    {
      unsigned t = std::thread::hardware_concurrency();
      if (const char *e = std::getenv("LLAMAR_N_THREADS")) {
        int v = std::atoi(e);
        if (v > 0) t = (unsigned)v;
      }
      cparams.n_threads = std::max(1u, t);
    }
    cparams.n_threads_batch = cparams.n_threads;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
      llama_model_free(model);
      Rcpp::stop("Failed to create llama context");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab     = llama_vocab_n_tokens(vocab);

    // Tokenize prompt
    std::vector<llama_token> tokens;
    tokens.resize(std::max<int>(32, prompt.size() + 8));
    int32_t ntok = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(), tokens.data(), (int32_t)tokens.size(), /*add_special*/ true, /*parse_special*/ false);
    if (ntok < 0) {
      tokens.resize(-ntok);
      ntok = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(), tokens.data(), (int32_t)tokens.size(), /*add_special*/ true, /*parse_special*/ false);
    }
    tokens.resize(ntok);

    // Feed prompt
    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
    int32_t rc = llama_decode(ctx, batch);
    if (rc < 0) {
      llama_free(ctx);
      llama_model_free(model);
      Rcpp::stop(std::string("llama_decode failed on prompt (rc=") + std::to_string(rc) + ")");
    }

    // Build sampler chain
    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    struct llama_sampler * chain = llama_sampler_chain_init(chain_params);
    // penalties first
    if (repeat_penalty > 1.0 || repeat_last_n != 0) {
      int lastn = repeat_last_n == 0 ? 64 : repeat_last_n;
      struct llama_sampler * pen = llama_sampler_init_penalties(lastn, (float)repeat_penalty, /*alpha_frequency*/0.0f, /*alpha_presence*/0.0f);
      llama_sampler_chain_add(chain, pen);
    }
    // top-k / top-p
    if (top_k > 0) {
      llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
    }
    if (top_p > 0.0 && top_p < 1.0) {
      llama_sampler_chain_add(chain, llama_sampler_init_top_p((float)top_p, /*min_keep*/1));
    }
    // temperature
    if (temperature > 0.0) {
      llama_sampler_chain_add(chain, llama_sampler_init_temp((float)temperature));
    } else {
      // greedy
      llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    }
    // distribution / RNG
    uint32_t sseed = seed == 0 ? LLAMA_DEFAULT_SEED : (uint32_t)seed;
    llama_sampler_chain_add(chain, llama_sampler_init_dist(sseed));

    std::string generated;
    generated.reserve(1024);
    std::vector<char> piece(4096);

    for (int i = 0; i < n_predict; ++i) {
      // sample next token using chain (it reads current logits from ctx)
      llama_token new_id = llama_sampler_sample(chain, ctx, /*idx*/0);

      // accept to update internal state
      llama_sampler_accept(chain, new_id);

      // decode this token
      llama_batch b = llama_batch_get_one(&new_id, 1);
      if (b.logits) b.logits[0] = 1;
      rc = llama_decode(ctx, b);
      if (rc < 0) break;

      // append and detokenize piece
      int32_t n = llama_token_to_piece(vocab, new_id, piece.data(), (int32_t)piece.size(), /*lstrip*/ 0, /*special*/ true);
      if (n > 0) {
        generated.append(piece.data(), piece.data() + n);
      } else if (n == (int32_t)piece.size()) {
        piece.resize(piece.size() * 2);
        n = llama_token_to_piece(vocab, new_id, piece.data(), (int32_t)piece.size(), 0, true);
        if (n > 0) {
          generated.append(piece.data(), piece.data() + n);
        }
      }

      // push token to history and check termination
      tokens.push_back(new_id);
      if (llama_vocab_is_eog(vocab, new_id)) break;
      if (!stops.empty()) {
        for (const auto & s : stops) {
          if (!s.empty() && generated.size() >= s.size()) {
            if (generated.compare(generated.size() - s.size(), s.size(), s) == 0) {
              // trim the stop sequence suffix and stop
              generated.resize(generated.size() - s.size());
              i = n_predict; // force loop end
              break;
            }
          }
        }
      }
    }

    // cleanup
    llama_sampler_free(chain);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return Rcpp::wrap(generated);
  } catch (std::exception &e) {
    Rcpp::stop(std::string("llama_generate_sampled error: ") + e.what());
  } catch (...) {
    Rcpp::stop("llama_generate_sampled: unknown error");
  }
}

// Apply model's chat template to role-tagged messages
SEXP llama_chat_format(SEXP model_path_, SEXP roles_, SEXP contents_, SEXP tmpl_, SEXP add_assistant_) {
  try {
    std::string model_path = as<std::string>(model_path_);
    CharacterVector roles(roles_);
    CharacterVector contents(contents_);
    bool add_assistant = as<bool>(add_assistant_);
    std::string tmpl;
    if (!Rf_isNull(tmpl_)) tmpl = as<std::string>(tmpl_);

    if (model_path.empty()) {
      Rcpp::stop("Model path is empty");
    }

    if (roles.size() != contents.size()) {
      Rcpp::stop("roles and contents must have same length");
    }

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.use_mmap     = true;
    mparams.use_mlock    = false;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
      Rcpp::stop(std::string("Failed to load model: ") + model_path);
    }

    // Build C array of messages
    std::vector<llama_chat_message> msgs;
    msgs.reserve(roles.size());
    std::vector<std::string> role_buf, content_buf;
    role_buf.reserve(roles.size());
    content_buf.reserve(roles.size());
    for (int i = 0; i < roles.size(); ++i) {
      if (roles[i] == NA_STRING || contents[i] == NA_STRING) {
        Rcpp::stop("roles/contents cannot be NA");
      }
      role_buf.emplace_back(as<std::string>(roles[i]));
      content_buf.emplace_back(as<std::string>(contents[i]));
      llama_chat_message m{ role_buf.back().c_str(), content_buf.back().c_str() };
      msgs.push_back(m);
    }

    // Buffer for output
    std::string out;
    out.resize(4096);

    const char * tmpl_c = tmpl.empty() ? nullptr : tmpl.c_str();
    int32_t need = llama_chat_apply_template(tmpl_c, msgs.data(), (size_t)msgs.size(), add_assistant, out.data(), (int32_t)out.size());
    if (need > (int32_t)out.size()) {
      out.resize(need);
      int32_t need2 = llama_chat_apply_template(tmpl_c, msgs.data(), (size_t)msgs.size(), add_assistant, out.data(), (int32_t)out.size());
      if (need2 <= 0) {
        llama_model_free(model);
        Rcpp::stop("chat template application failed");
      }
      out.resize(need2);
    } else if (need <= 0) {
      llama_model_free(model);
      Rcpp::stop("chat template application failed");
    } else {
      out.resize(need);
    }

    llama_model_free(model);
    llama_backend_free();
    return Rcpp::wrap(out);
  } catch (std::exception &e) {
    Rcpp::stop(std::string("llama_chat_format error: ") + e.what());
  } catch (...) {
    Rcpp::stop("llama_chat_format: unknown error");
  }
}
