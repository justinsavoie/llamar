// interface.cpp — CPU-only, guarded R bridge for llama.cpp

#include <Rcpp.h>
#include "llama.h"

#include <thread>
#include <random>
#include <string>
#include <vector>
#include <algorithm>

using namespace Rcpp;

extern "C" SEXP llama_build_test();
extern "C" SEXP llama_generate_greedy(SEXP model_path_, SEXP prompt_, SEXP n_predict_, SEXP n_ctx_);
extern "C" SEXP llama_generate_sampled(SEXP model_path_, SEXP prompt_, SEXP n_predict_, SEXP n_ctx_,
                                        SEXP temperature_, SEXP top_p_, SEXP top_k_,
                                        SEXP repeat_penalty_, SEXP repeat_last_n_, SEXP seed_, SEXP stop_);
extern "C" SEXP llama_chat_format(SEXP model_path_, SEXP roles_, SEXP contents_, SEXP tmpl_, SEXP add_assistant_);

// --- tiny helpers ------------------------------------------------------------

static unsigned env_threads_default() {
  unsigned t = std::thread::hardware_concurrency();
  if (const char *e = std::getenv("LLAMAR_N_THREADS")) {
    int v = std::atoi(e);
    if (v > 0) t = (unsigned) v;
  }
  return std::max(1u, t);
}

static bool env_use_mmap_default() {
  bool use_mmap = true;
  if (const char *e = std::getenv("LLAMAR_USE_MMAP")) {
    use_mmap = std::atoi(e) != 0;
  }
  return use_mmap;
}

SEXP llama_build_test() {
  return Rf_mkString("Success! R package can see llama.cpp headers.");
}

// --- GREEDY ------------------------------------------------------------------

SEXP llama_generate_greedy(SEXP model_path_, SEXP prompt_, SEXP n_predict_, SEXP n_ctx_) {
  try {
    std::string model_path = as<std::string>(model_path_);
    std::string prompt     = as<std::string>(prompt_);
    int n_predict          = as<int>(n_predict_);
    int n_ctx              = as<int>(n_ctx_);

    if (model_path.empty()) Rcpp::stop("Model path is empty");
    if (n_predict <= 0)     return Rf_mkString("");

    // init backend
    llama_backend_init();

    // model params (CPU only)
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.use_mmap     = env_use_mmap_default();
    mparams.use_mlock    = false;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
      llama_backend_free();
      Rcpp::stop(std::string("Failed to load model: ") + model_path);
    }

    // context params (CPU only)
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx            = std::max(8, n_ctx <= 0 ? 512 : n_ctx);
    cparams.offload_kqv      = false;
    cparams.op_offload       = false;
    cparams.n_threads        = env_threads_default();
    cparams.n_threads_batch  = cparams.n_threads;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
      llama_model_free(model);
      llama_backend_free();
      Rcpp::stop("Failed to create llama context");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (!vocab) {
      llama_free(ctx);
      llama_model_free(model);
      llama_backend_free();
      Rcpp::stop("Null vocab pointer from model");
    }
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0) {
      llama_free(ctx);
      llama_model_free(model);
      llama_backend_free();
      Rcpp::stop("Invalid vocabulary size from model");
    }

    // tokenize
    std::vector<llama_token> tokens;
    tokens.resize(std::max<int>(32, (int)prompt.size() + 8));
    int32_t ntok = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(),
                                  tokens.data(), (int32_t)tokens.size(),
                                  /*add_special=*/true, /*parse_special=*/false);
    if (ntok < 0) {
      tokens.resize(-ntok);
      ntok = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(),
                            tokens.data(), (int32_t)tokens.size(),
                            true, false);
    }
    tokens.resize(ntok);

    // feed prompt
    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
    int32_t rc = llama_decode(ctx, batch);
    if (rc < 0) {
      llama_free(ctx);
      llama_model_free(model);
      llama_backend_free();
      Rcpp::stop(std::string("llama_decode failed on prompt (rc=") + std::to_string(rc) + ")");
    }

    // greedy loop
    std::string generated;
    generated.reserve(1024);
    std::vector<char> piece(4096);

    for (int i = 0; i < n_predict; ++i) {
      llama_token last = tokens.back();
      llama_batch b = llama_batch_get_one(&last, 1);
      if (b.logits) b.logits[0] = 1;      // request logits

      rc = llama_decode(ctx, b);
      if (rc < 0) break;

      float * logits = llama_get_logits(ctx);
      if (!logits) break;

      int best_id = 0;
      float best_logit = logits[0];
      for (int vid = 1; vid < n_vocab; ++vid) {
        if (logits[vid] > best_logit) { best_logit = logits[vid]; best_id = vid; }
      }

      tokens.push_back((llama_token)best_id);

      if (llama_vocab_is_eog(vocab, (llama_token)best_id)) break;

      int32_t n = llama_token_to_piece(vocab, (llama_token)best_id,
                                       piece.data(), (int32_t)piece.size(),
                                       /*lstrip=*/0, /*special=*/true);
      if (n > 0) {
        generated.append(piece.data(), piece.data() + n);
      } else if (n == (int32_t)piece.size()) {
        piece.resize(piece.size() * 2);
        n = llama_token_to_piece(vocab, (llama_token)best_id, piece.data(),
                                 (int32_t)piece.size(), 0, true);
        if (n > 0) generated.append(piece.data(), piece.data() + n);
      }
    }

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

// --- SAMPLED (top-k/top-p/temp/penalties) -----------------------------------

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

    if (model_path.empty()) Rcpp::stop("Model path is empty");
    if (n_predict <= 0)     return Rf_mkString("");

    // sanitize sampling params
    if (!(temperature > 0.0)) temperature = 1.0;
    if (!(top_p > 0.0 && top_p <= 1.0)) top_p = 1.0;
    if (top_k <= 0) top_k = 1;
    if (!(repeat_penalty > 0.0)) repeat_penalty = 1.0;
    if (repeat_last_n < 0) repeat_last_n = 0;

    llama_backend_init();

    // model params (CPU only)
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.use_mmap     = env_use_mmap_default();
    mparams.use_mlock    = false;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
      llama_backend_free();
      Rcpp::stop(std::string("Failed to load model: ") + model_path);
    }

    // context params (CPU only)
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx            = std::max(8, n_ctx <= 0 ? 512 : n_ctx);
    cparams.offload_kqv      = false;
    cparams.op_offload       = false;
    cparams.n_threads        = env_threads_default();
    cparams.n_threads_batch  = cparams.n_threads;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
      llama_model_free(model);
      llama_backend_free();
      Rcpp::stop("Failed to create llama context");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (!vocab) {
      llama_free(ctx);
      llama_model_free(model);
      llama_backend_free();
      Rcpp::stop("Null vocab pointer from model");
    }
    int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0) {
      llama_free(ctx);
      llama_model_free(model);
      llama_backend_free();
      Rcpp::stop("Invalid vocabulary size from model");
    }

    // tokenize
    std::vector<llama_token> tokens;
    tokens.resize(std::max<int>(32, (int)prompt.size() + 8));
    int32_t ntok = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(),
                                  tokens.data(), (int32_t)tokens.size(),
                                  /*add_special=*/true, /*parse_special=*/false);
    if (ntok < 0) {
      tokens.resize(-ntok);
      ntok = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(),
                            tokens.data(), (int32_t)tokens.size(),
                            true, false);
    }
    tokens.resize(ntok);

    // feed prompt (single batch)
    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
    int32_t rc = llama_decode(ctx, batch);
    if (rc < 0) {
      llama_free(ctx);
      llama_model_free(model);
      llama_backend_free();
      Rcpp::stop(std::string("llama_decode failed on prompt (rc=") + std::to_string(rc) + ")");
    }

    // 🔧 PRIME LOGITS: request logits on the last prompt token so the first sample has data
    {
      llama_token last_tok = tokens.back();
      llama_batch lastb = llama_batch_get_one(&last_tok, 1);
      if (lastb.logits) lastb.logits[0] = 1;
      rc = llama_decode(ctx, lastb);
      if (rc < 0) {
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        Rcpp::stop("llama_decode failed while priming logits after prompt");
      }
    }

    // build sampler chain (defensive: check each sampler)
    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    struct llama_sampler * chain = llama_sampler_chain_init(chain_params);
    if (!chain) {
      llama_free(ctx);
      llama_model_free(model);
      llama_backend_free();
      Rcpp::stop("Failed to initialize sampler chain");
    }

    auto add_sampler = [&](llama_sampler *s, const char *name) {
      if (!s) {
        llama_sampler_free(chain);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        Rcpp::stop(std::string("Failed to init sampler: ") + name);
      }
      llama_sampler_chain_add(chain, s);
    };

    // repetition penalties (only if meaningful)
    if (repeat_penalty != 1.0 || repeat_last_n > 0) {
      int lastn = repeat_last_n == 0 ? 64 : repeat_last_n;
      lastn = std::max(0, std::min(lastn, static_cast<int>(cparams.n_ctx)));
      add_sampler(llama_sampler_init_penalties(lastn, (float)repeat_penalty,
                                               /*alpha_frequency*/0.0f,
                                               /*alpha_presence*/0.0f),
                  "penalties");
    }

    // top-k / top-p
    if (top_k > n_vocab) top_k = n_vocab;
    if (top_k > 1) add_sampler(llama_sampler_init_top_k(top_k), "top_k");
    if (top_p > 0.0 && top_p < 1.0) add_sampler(llama_sampler_init_top_p((float)top_p, /*min_keep*/1), "top_p");

    // temperature or greedy
    if (temperature > 0.0) add_sampler(llama_sampler_init_temp((float)temperature), "temp");
    else                   add_sampler(llama_sampler_init_greedy(), "greedy");

    // RNG / distribution
    uint32_t sseed = (seed == 0) ? LLAMA_DEFAULT_SEED : (uint32_t) seed;
    add_sampler(llama_sampler_init_dist(sseed), "dist");

    // generation loop
    std::string generated;
    generated.reserve(1024);
    std::vector<char> piece(4096);

    for (int i = 0; i < n_predict; ++i) {
      // guard: logits must be present before sampling
      const float * logits_ptr = llama_get_logits(ctx);
      if (!logits_ptr) {
        llama_sampler_free(chain);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        Rcpp::stop("Null logits before sampling (missing logits request?)");
      }

      llama_token new_id = llama_sampler_sample(chain, ctx, /*idx*/0);
      if (new_id < 0 || new_id >= n_vocab) {
        llama_sampler_free(chain);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        Rcpp::stop("Invalid token id sampled");
      }

      // update sampler internal state
      llama_sampler_accept(chain, new_id);

      // decode next step, requesting logits for the newly generated token
      llama_batch b = llama_batch_get_one(&new_id, 1);
      if (b.logits) b.logits[0] = 1;
      rc = llama_decode(ctx, b);
      if (rc < 0) {
        llama_sampler_free(chain);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        Rcpp::stop("llama_decode failed during generation");
      }

      // append text
      int32_t n = llama_token_to_piece(vocab, new_id, piece.data(), (int32_t)piece.size(),
                                       /*lstrip=*/0, /*special=*/true);
      if (n > 0) {
        generated.append(piece.data(), piece.data() + n);
      } else if (n == (int32_t)piece.size()) {
        piece.resize(piece.size() * 2);
        n = llama_token_to_piece(vocab, new_id, piece.data(), (int32_t)piece.size(), 0, true);
        if (n > 0) generated.append(piece.data(), piece.data() + n);
      }

      // track history & stop conditions
      tokens.push_back(new_id);
      if (llama_vocab_is_eog(vocab, new_id)) break;

      if (!stops.empty()) {
        for (const auto & s : stops) {
          if (!s.empty() && generated.size() >= s.size()) {
            if (generated.compare(generated.size() - s.size(), s.size(), s) == 0) {
              generated.resize(generated.size() - s.size()); // trim stop sequence
              i = n_predict; // break outer loop
              break;
            }
          }
        }
      }
    }

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

// --- CHAT TEMPLATE -----------------------------------------------------------

SEXP llama_chat_format(SEXP model_path_, SEXP roles_, SEXP contents_, SEXP tmpl_, SEXP add_assistant_) {
  try {
    std::string model_path = as<std::string>(model_path_);
    CharacterVector roles(roles_);
    CharacterVector contents(contents_);
    bool add_assistant = as<bool>(add_assistant_);
    std::string tmpl;
    if (!Rf_isNull(tmpl_)) tmpl = as<std::string>(tmpl_);

    if (model_path.empty()) Rcpp::stop("Model path is empty");
    if (roles.size() != contents.size()) Rcpp::stop("roles and contents must have same length");

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.use_mmap     = true;
    mparams.use_mlock    = false;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
      llama_backend_free();
      Rcpp::stop(std::string("Failed to load model: ") + model_path);
    }

    std::vector<llama_chat_message> msgs;
    msgs.reserve(roles.size());
    std::vector<std::string> role_buf, content_buf;
    role_buf.reserve(roles.size());
    content_buf.reserve(roles.size());
    for (int i = 0; i < roles.size(); ++i) {
      if (roles[i] == NA_STRING || contents[i] == NA_STRING) {
        llama_model_free(model);
        llama_backend_free();
        Rcpp::stop("roles/contents cannot be NA");
      }
      role_buf.emplace_back(as<std::string>(roles[i]));
      content_buf.emplace_back(as<std::string>(contents[i]));
      llama_chat_message m{ role_buf.back().c_str(), content_buf.back().c_str() };
      msgs.push_back(m);
    }

    std::string out;
    out.resize(4096);

    const char * tmpl_c = tmpl.empty() ? nullptr : tmpl.c_str();
    int32_t need = llama_chat_apply_template(tmpl_c, msgs.data(), (size_t)msgs.size(),
                                             add_assistant, out.data(), (int32_t)out.size());
    if (need > (int32_t)out.size()) {
      out.resize(need);
      int32_t need2 = llama_chat_apply_template(tmpl_c, msgs.data(), (size_t)msgs.size(),
                                                add_assistant, out.data(), (int32_t)out.size());
      if (need2 <= 0) {
        llama_model_free(model);
        llama_backend_free();
        Rcpp::stop("chat template application failed");
      }
      out.resize(need2);
    } else if (need <= 0) {
      llama_model_free(model);
      llama_backend_free();
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
