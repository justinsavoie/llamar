#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

// Forward declaration of our .Call routine
extern SEXP llama_build_test(void);
extern SEXP llama_generate_greedy(SEXP, SEXP, SEXP, SEXP);
extern SEXP llama_generate_sampled(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP llama_chat_format(SEXP, SEXP, SEXP, SEXP, SEXP);

static const R_CallMethodDef CallEntries[] = {
    {"llama_build_test", (DL_FUNC) &llama_build_test, 0},
    {"llama_generate_greedy", (DL_FUNC) &llama_generate_greedy, 4},
    {"llama_generate_sampled", (DL_FUNC) &llama_generate_sampled, 11},
    {"llama_chat_format", (DL_FUNC) &llama_chat_format, 5},
    {NULL, NULL, 0}
};

void R_init_llamar(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
