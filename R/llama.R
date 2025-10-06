llama_build_test <- function() .Call("llama_build_test")

#' Generate text using llama.cpp (CPU-only, greedy)
#'
#' @param model Path to a GGUF model file
#' @param prompt Prompt string
#' @param n_predict Number of tokens to generate
#' @param n_ctx Context length (smaller uses less memory)
#' @return Generated continuation as a character scalar
#' @export
llama_generate_greedy <- function(model, prompt, n_predict = 64L, n_ctx = 512L) {
  stopifnot(is.character(model), length(model) == 1L)
  stopifnot(is.character(prompt), length(prompt) == 1L)
  if (!nzchar(model)) stop("model path is empty; provide a GGUF file path", call. = FALSE)
  model <- path.expand(model)
  if (!file.exists(model)) stop(sprintf("model file not found: %s", model), call. = FALSE)
  n_predict <- as.integer(n_predict)
  n_ctx <- as.integer(n_ctx)
  if (is.na(n_predict) || n_predict < 0L) stop("n_predict must be a non-negative integer", call. = FALSE)
  if (is.na(n_ctx) || n_ctx <= 0L) stop("n_ctx must be a positive integer", call. = FALSE)
  .Call("llama_generate_greedy", model, prompt, n_predict, n_ctx)
}

#' Generate text with sampling controls (temperature, top-p/k, repetition)
#'
#' @param model Path to a GGUF model file
#' @param prompt Prompt string (already formatted for chat if needed)
#' @param n_predict Number of tokens to generate
#' @param n_ctx Context length
#' @param temperature Temperature (>0 for sampling; 0 for greedy)
#' @param top_p Nucleus sampling probability (0..1)
#' @param top_k Top-k cutoff (integer > 0)
#' @param repeat_penalty Repetition penalty (>1 penalizes repeats)
#' @param repeat_last_n Window for repetition penalty
#' @param seed RNG seed (0 for default)
#' @param stop Optional character vector of stop sequences
#' @export
llama_generate <- function(model, prompt, n_predict = 64L, n_ctx = 512L,
                           temperature = 0.8, top_p = 0.95, top_k = 40L,
                           repeat_penalty = 1.0, repeat_last_n = 64L,
                           seed = 0L, stop = character()) {
  stopifnot(is.character(model), length(model) == 1L)
  stopifnot(is.character(prompt), length(prompt) == 1L)
  if (!nzchar(model)) stop("model path is empty; provide a GGUF file path", call. = FALSE)
  model <- path.expand(model)
  if (!file.exists(model)) stop(sprintf("model file not found: %s", model), call. = FALSE)
  n_predict <- as.integer(n_predict)
  n_ctx <- as.integer(n_ctx)
  top_k <- as.integer(top_k)
  repeat_last_n <- as.integer(repeat_last_n)
  seed <- as.integer(seed)
  if (is.na(n_predict) || n_predict < 0L) stop("n_predict must be a non-negative integer", call. = FALSE)
  if (is.na(n_ctx) || n_ctx <= 0L) stop("n_ctx must be a positive integer", call. = FALSE)
  .Call("llama_generate_sampled", model, prompt, n_predict, n_ctx,
        as.numeric(temperature), as.numeric(top_p), top_k,
        as.numeric(repeat_penalty), repeat_last_n, seed, as.character(stop))
}

#' Format chat messages using the model's chat template
#'
#' @param model Path to a GGUF model file
#' @param messages A list of lists with elements role and content
#' @param template Optional template string to override model default
#' @param add_assistant Whether to append an assistant prefix for generation
#' @return A single prompt string suitable for llama_generate*
#' @export
chat_format <- function(model, messages, template = NULL, add_assistant = TRUE) {
  stopifnot(is.character(model), length(model) == 1L)
  stopifnot(is.list(messages))
  if (!nzchar(model)) stop("model path is empty; provide a GGUF file path", call. = FALSE)
  model <- path.expand(model)
  if (!file.exists(model)) stop(sprintf("model file not found: %s", model), call. = FALSE)
  roles <- vapply(messages, function(x) x[["role"]], character(1), USE.NAMES = FALSE)
  contents <- vapply(messages, function(x) x[["content"]], character(1), USE.NAMES = FALSE)
  .Call("llama_chat_format", model, as.character(roles), as.character(contents),
        if (is.null(template)) NULL else as.character(template), as.logical(add_assistant))
}

#' Convenience helper: format chat and generate text
#'
#' @inheritParams chat_format
#' @inheritParams llama_generate
#' @export
chat <- function(model, messages, n_predict = 64L, n_ctx = 512L,
                 temperature = 0.8, top_p = 0.95, top_k = 40L,
                 repeat_penalty = 1.0, repeat_last_n = 64L,
                 seed = 0L, stop = character(), template = NULL,
                 add_assistant = TRUE) {
  prompt <- chat_format(model, messages, template = template, add_assistant = add_assistant)
  llama_generate(model, prompt, n_predict = n_predict, n_ctx = n_ctx,
                 temperature = temperature, top_p = top_p, top_k = top_k,
                 repeat_penalty = repeat_penalty, repeat_last_n = repeat_last_n,
                 seed = seed, stop = stop)
}
