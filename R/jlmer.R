#' Fit mixed-effects models with Julia MixedModels
#'
#' `jlmer()` provides an [lme4::lmer()]/[lme4::glmer()]-style interface to
#' Julia's MixedModels package. The model is fitted in Julia and converted by
#' JellyMe4 to an R `lmerMod` (Gaussian) or `glmerMod` (binomial or Poisson)
#' object. The returned object therefore works with most software that supports
#' `lme4` models, including `emmeans`, `broom.mixed`, and `ggeffects`.
#'
#' @param formula A two-sided mixed-model formula. It must contain at least one
#'   random-effects term, such as `(1 | subject)`.
#' @param data A data frame containing every variable used in `formula`.
#' @param REML Logical scalar. For Gaussian models, use restricted maximum
#'   likelihood when `TRUE` and maximum likelihood when `FALSE`. Generalized
#'   models are always fitted by maximum likelihood; an explicitly supplied
#'   `REML = TRUE` is ignored with a warning.
#' @param JULIA_HOME Optional path to the Julia installation, passed to
#'   [JuliaCall::julia_setup()]. When `NULL`, JuliaCall searches for Julia.
#' @param family An R family object, family function, or one of `"gaussian"`,
#'   `"binomial"`, or `"poisson"`. Links supported by the corresponding R
#'   family constructor can be requested in the usual way, for example
#'   `binomial(link = "probit")`. JellyMe4 currently supports conversion of
#'   Gaussian identity-link models, Bernoulli/binomial models, and Poisson
#'   models. Quasi, Gamma, inverse-Gaussian, negative-binomial, and other
#'   families cannot be converted to an `lme4` object and are rejected before
#'   Julia is started.
#' @param weights Optional numeric case weights, or a single character string
#'   naming a column in `data`. For a binomial model, supplying weights selects
#'   Julia's `Binomial()` distribution: the response must be a proportion and
#'   the weights are the numbers of trials. Without weights, binomial models
#'   use `Bernoulli()` and require a binary response. The `cbind(success,
#'   failure)` response syntax is not supported by JellyMe4; use a proportion
#'   plus weights instead.
#' @param na.action A missing-data function or its name. The default,
#'   [stats::na.omit()], removes rows missing any formula variable or weight
#'   before transferring data to Julia. [stats::na.exclude()] also restores
#'   excluded positions in compatible residual and prediction methods. Use
#'   `NULL` or [stats::na.pass()] to send missing values to Julia unchanged.
#' @param lmer_test Logical scalar. If `TRUE`, a Gaussian result is converted
#'   with `lmerTest::as_lmerModLmerTest()`. This is unavailable for generalized
#'   models and requires the suggested R package `lmerTest`.
#' @param lmer_test_tol Non-negative numeric tolerance passed to
#'   `lmerTest::as_lmerModLmerTest()`.
#' @param julia_setup_args Named list of additional arguments passed to
#'   [JuliaCall::julia_setup()]. Supply `JULIA_HOME` through its dedicated
#'   argument instead.
#' @param julia_packages Character vector of additional Julia packages to load.
#'   `MixedModels`, `RCall`, and `JellyMe4` are always loaded. Link types are
#'   accessed through MixedModels' GLM dependency, so a separate direct Julia
#'   installation of GLM is not required.
#' @param fit_args Named list of additional Julia `fit()` keyword arguments.
#'   Common generalized-model options include `nAGQ` and `fast`. Values are
#'   transferred as Julia variables rather than interpolated into code.
#' @param keep_julia_model Logical scalar. If `TRUE`, retain the fitted model in
#'   Julia's `Main` module and record its variable name in `attr(x, "jlmer")`.
#'   Data, formula, and argument variables are still cleared.
#' @param verbose Logical scalar. If `TRUE`, print the generated Julia fit call.
#' @param ... Additional *named* Julia `fit()` keyword arguments. These are
#'   combined with `fit_args`; duplicate names are an error.
#'
#' @return An `lmerMod` for a Gaussian model or a `glmerMod` for a binomial or
#'   Poisson model. Gaussian models can optionally be returned as
#'   `lmerModLmerTest`. Backend details, including the family, link, omitted
#'   rows, and Julia model variable (when retained), are stored in
#'   `attr(result, "jlmer")`.
#'
#' @details
#' Julia packages `MixedModels`, `RCall`, and `JellyMe4` must be installed in
#' the active Julia environment. `jlmer()` does not install or update Julia
#' packages. The first Julia fit in an R session can be noticeably slower due
#' to Julia's just-in-time compilation.
#'
#' MixedModels uses `Bernoulli()` for an unaggregated 0/1 response and
#' `Binomial()` for a response expressed as successes/trials with the trial
#' counts supplied as case weights. This differs slightly from R, where both
#' representations are called `binomial`.
#'
#' JellyMe4 warns that some numerical accuracy is lost while reconstructing a
#' `glmerMod`. The Julia fit supplies the estimates, but inferential workflows
#' should still compare important results with a native fit when feasible.
#'
#' @examples
#' \dontrun{
#' # Gaussian LMM (REML by default)
#' fit_gaussian <- jlmer(
#'   Reaction ~ Days + (Days | Subject),
#'   data = lme4::sleepstudy
#' )
#' summary(fit_gaussian)
#'
#' # Binary logistic GLMM. A noncanonical link can be supplied as, for example,
#' # family = binomial(link = "probit").
#' set.seed(1)
#' binary_data <- transform(
#'   lme4::sleepstudy,
#'   responded = rbinom(nrow(lme4::sleepstudy), 1, plogis(-1 + Days / 5))
#' )
#' fit_binary <- jlmer(
#'   responded ~ Days + (1 | Subject),
#'   data = binary_data,
#'   family = binomial()
#' )
#'
#' # Aggregated binomial GLMM: response is a proportion and weights are trials.
#' cbpp <- transform(lme4::cbpp, proportion = incidence / size)
#' fit_binomial <- jlmer(
#'   proportion ~ period + (1 | herd),
#'   data = cbpp,
#'   family = binomial(),
#'   weights = "size",
#'   nAGQ = 9
#' )
#'
#' # Poisson log-link GLMM for counts.
#' data("grouseticks", package = "lme4")
#' fit_poisson <- jlmer(
#'   TICKS ~ YEAR + cHEIGHT + (1 | BROOD) + (1 | LOCATION),
#'   data = grouseticks,
#'   family = poisson(),
#'   fast = TRUE
#' )
#' }
#'
#' @md
#' @export
jlmer <- function(formula,
                  data,
                  REML = TRUE,
                  JULIA_HOME = NULL,
                  family = stats::gaussian(),
                  weights = NULL,
                  na.action = stats::na.omit,
                  lmer_test = FALSE,
                  lmer_test_tol = 1e-8,
                  julia_setup_args = list(),
                  julia_packages = character(),
                  fit_args = list(),
                  keep_julia_model = FALSE,
                  verbose = FALSE,
                  ...) {
  reml_was_supplied <- !missing(REML)
  original_call <- match.call()

  if (!requireNamespace("JuliaCall", quietly = TRUE)) {
    stop("Package \"JuliaCall\" must be installed to use jlmer().", call. = FALSE)
  }

  if (is.character(formula)) {
    if (length(formula) != 1L) {
      stop("A character formula must have length one.", call. = FALSE)
    }
    formula <- stats::as.formula(formula, env = parent.frame())
  }
  checkmate::assert_formula(formula)
  if (length(formula) != 3L) {
    stop("formula must be a two-sided mixed-model formula.", call. = FALSE)
  }
  if (!.jlmer_has_random_effect(formula)) {
    stop("formula must contain at least one random-effects term, such as (1 | subject).", call. = FALSE)
  }

  checkmate::assert_data_frame(data, min.rows = 1L)
  data <- as.data.frame(data)
  n_original <- nrow(data)
  original_rownames <- rownames(data)

  checkmate::assert_flag(REML)
  checkmate::assert_string(JULIA_HOME, null.ok = TRUE)
  checkmate::assert_flag(lmer_test)
  checkmate::assert_number(lmer_test_tol, lower = 0, finite = TRUE)
  checkmate::assert_list(julia_setup_args, names = "unique")
  checkmate::assert_character(julia_packages, any.missing = FALSE)
  checkmate::assert_list(fit_args, names = "unique")
  checkmate::assert_flag(keep_julia_model)
  checkmate::assert_flag(verbose)

  .jlmer_assert_named_list(julia_setup_args, "julia_setup_args")
  .jlmer_assert_named_list(fit_args, "fit_args")

  if ("JULIA_HOME" %in% names(julia_setup_args)) {
    stop("Supply JULIA_HOME through the JULIA_HOME argument, not julia_setup_args.", call. = FALSE)
  }

  family_info <- .jlmer_family(family, weighted = !is.null(weights))
  is_gaussian <- identical(family_info$family, "gaussian")
  if (!is_gaussian) {
    if (reml_was_supplied && isTRUE(REML)) {
      warning("REML is not defined for generalized mixed models; fitting by maximum likelihood.", call. = FALSE)
    }
    REML <- FALSE
    if (isTRUE(lmer_test)) {
      stop("lmer_test = TRUE is only available for Gaussian models.", call. = FALSE)
    }
  }

  dot_args <- list(...)
  if (length(dot_args) > 0L &&
      (is.null(names(dot_args)) || anyNA(names(dot_args)) || any(names(dot_args) == ""))) {
    stop("Additional arguments in ... must be named Julia fit() keyword arguments.", call. = FALSE)
  }
  if (length(dot_args) > 0L) {
    fit_args <- c(fit_args, dot_args)
  }
  if (anyDuplicated(names(fit_args))) {
    stop("Duplicate Julia fit() keyword arguments were supplied.", call. = FALSE)
  }
  reserved_fit_args <- intersect(names(fit_args), c("REML", "weights"))
  if (length(reserved_fit_args) > 0L) {
    stop(
      "Supply ", paste(reserved_fit_args, collapse = " and "),
      " through the corresponding jlmer() argument instead of fit_args or the dots.",
      call. = FALSE
    )
  }
  if (length(fit_args) > 0L) {
    valid_julia_names <- grepl("^[A-Za-z_][A-Za-z0-9_]*$", names(fit_args))
    if (!all(valid_julia_names)) {
      stop(
        "All Julia fit() keyword arguments must be valid Julia identifiers. Invalid: ",
        paste(names(fit_args)[!valid_julia_names], collapse = ", "),
        call. = FALSE
      )
    }
  }

  required_packages <- c("MixedModels", "RCall", "JellyMe4")
  julia_packages <- unique(c(required_packages, julia_packages))
  valid_package_names <- grepl(
    "^[A-Za-z_][A-Za-z0-9_]*(\\.[A-Za-z_][A-Za-z0-9_]*)*$",
    julia_packages
  )
  if (!all(valid_package_names)) {
    stop(
      "Invalid Julia package name(s): ",
      paste(julia_packages[!valid_package_names], collapse = ", "),
      call. = FALSE
    )
  }

  formula_terms <- tryCatch(
    stats::terms(formula, data = data),
    error = function(e) {
      stop("Unable to parse formula: ", conditionMessage(e), call. = FALSE)
    }
  )
  model_vars <- all.vars(formula_terms)
  missing_vars <- setdiff(model_vars, names(data))
  if (length(missing_vars) > 0L) {
    stop("Variables not found in data: ", paste(missing_vars, collapse = ", "), call. = FALSE)
  }

  resolved_weights <- .jlmer_weights(weights, data)
  model_data <- data[, model_vars, drop = FALSE]
  if (!is.null(resolved_weights)) {
    model_data[["..jlmer_weights"]] <- resolved_weights
  }

  na_result <- .jlmer_apply_na_action(model_data, na.action, original_rownames)
  keep_rows <- na_result$keep
  omitted_rows <- na_result$omitted
  data <- data[keep_rows, , drop = FALSE]
  if (!is.null(resolved_weights)) {
    resolved_weights <- resolved_weights[keep_rows]
  }
  if (nrow(data) == 0L) {
    stop("No complete observations remain after applying na.action.", call. = FALSE)
  }

  response <- tryCatch(
    eval(formula[[2L]], envir = data, enclos = environment(formula)),
    error = function(e) {
      stop("Unable to evaluate the response in formula: ", conditionMessage(e), call. = FALSE)
    }
  )
  .jlmer_validate_response(response, family_info, resolved_weights, n_expected = nrow(data))

  user_fit_args <- fit_args
  if (!is.null(resolved_weights)) {
    fit_args <- c(list(weights = resolved_weights), fit_args)
  }

  julia_setup_call <- c(list(JULIA_HOME = JULIA_HOME), julia_setup_args)
  julia_setup_call <- julia_setup_call[!vapply(julia_setup_call, is.null, logical(1L))]
  tryCatch(
    do.call(JuliaCall::julia_setup, julia_setup_call),
    error = function(e) {
      stop("Unable to initialize Julia with JuliaCall::julia_setup(): ", conditionMessage(e), call. = FALSE)
    }
  )

  julia_using_call <- sprintf("using %s", paste(julia_packages, collapse = ", "))
  tryCatch(
    JuliaCall::julia_command(julia_using_call),
    error = function(e) {
      stop(
        "Unable to load required Julia package(s): ",
        paste(julia_packages, collapse = ", "),
        ". Install them in the active Julia environment before using jlmer(). Original error: ",
        conditionMessage(e),
        call. = FALSE
      )
    }
  )

  prefix <- gsub("[^A-Za-z0-9_]", "_", basename(tempfile("dependlab_jlmer_")))
  data_var <- paste0(prefix, "_data")
  formula_var <- paste0(prefix, "_formula")
  model_var <- paste0(prefix, "_model")
  fit_arg_vars <- character()

  cleanup_vars <- function() {
    vars <- c(data_var, formula_var, fit_arg_vars)
    if (!isTRUE(keep_julia_model)) {
      vars <- c(vars, model_var)
    }
    cleanup_call <- paste0(vars, " = nothing", collapse = "; ")
    try(JuliaCall::julia_command(paste0(cleanup_call, "; GC.gc()")), silent = TRUE)
  }
  on.exit(cleanup_vars(), add = TRUE)

  tryCatch(
    {
      JuliaCall::julia_assign(data_var, data)
      JuliaCall::julia_assign(formula_var, formula)
    },
    error = function(e) {
      stop("Unable to transfer data or formula to Julia: ", conditionMessage(e), call. = FALSE)
    }
  )

  fit_arg_expr <- character()
  if (length(fit_args) > 0L) {
    fit_arg_expr <- vapply(names(fit_args), function(arg_name) {
      arg_var <- paste0(prefix, "_arg_", arg_name)
      fit_arg_vars <<- c(fit_arg_vars, arg_var)
      tryCatch(
        JuliaCall::julia_assign(arg_var, fit_args[[arg_name]]),
        error = function(e) {
          stop(
            "Unable to transfer Julia fit() argument '", arg_name, "': ",
            conditionMessage(e),
            call. = FALSE
          )
        }
      )
      paste0(arg_name, "=", arg_var)
    }, character(1L))
  }

  positional_args <- c(formula_var, data_var)
  keyword_args <- fit_arg_expr
  if (is_gaussian) {
    keyword_args <- c(
      paste0("REML=", if (isTRUE(REML)) "true" else "false"),
      keyword_args
    )
  } else {
    positional_args <- c(
      positional_args,
      family_info$julia_distribution,
      family_info$julia_link
    )
  }
  fit_call <- sprintf(
    "%s = fit(MixedModel, %s%s);",
    model_var,
    paste(positional_args, collapse = ", "),
    if (length(keyword_args) > 0L) paste0("; ", paste(keyword_args, collapse = ", ")) else ""
  )
  if (isTRUE(verbose)) {
    message("Julia jlmer call: ", fit_call)
  }

  tryCatch(
    JuliaCall::julia_command(fit_call),
    error = function(e) {
      stop(
        "Julia MixedModels ", if (is_gaussian) "LMM" else "GLMM",
        " fit failed (family = ", family_info$family,
        ", link = ", family_info$link, "): ",
        conditionMessage(e),
        call. = FALSE
      )
    }
  )

  r_class <- if (is_gaussian) "lmerMod" else "glmerMod"
  model <- tryCatch(
    JuliaCall::julia_eval(
      sprintf("robject(:%s, (%s, %s));", r_class, model_var, data_var),
      need_return = "R"
    ),
    error = function(e) {
      stop(
        "JellyMe4 conversion to an R ", r_class, " failed: ",
        conditionMessage(e),
        call. = FALSE
      )
    }
  )

  if (isTRUE(lmer_test)) {
    if (!requireNamespace("lmerTest", quietly = TRUE)) {
      stop("Package \"lmerTest\" must be installed when lmer_test = TRUE.", call. = FALSE)
    }
    model <- tryCatch(
      lmerTest::as_lmerModLmerTest(model, tol = lmer_test_tol),
      error = function(e) {
        stop("lmerTest conversion failed: ", conditionMessage(e), call. = FALSE)
      }
    )
  }

  # JellyMe4 necessarily constructs its model with temporary R variable names.
  # Restore a reproducible call and the original missing-row action for methods
  # such as update(), residuals(), and predict().
  if (methods::is(model, "merMod")) {
    original_call[[1L]] <- quote(dependlab::jlmer)
    methods::slot(model, "call") <- original_call
    if (!is.null(na_result$action)) {
      model_frame <- methods::slot(model, "frame")
      attr(model_frame, "na.action") <- na_result$action
      methods::slot(model, "frame") <- model_frame
    }
  }

  attr(model, "jlmer") <- list(
    engine = "MixedModels.jl",
    model = if (is_gaussian) "linear" else "generalized",
    family = family_info$family,
    link = family_info$link,
    julia_distribution = family_info$julia_distribution,
    REML = if (is_gaussian) REML else FALSE,
    lmer_test = lmer_test,
    lmer_test_tol = if (isTRUE(lmer_test)) lmer_test_tol else NA_real_,
    julia_model_var = if (isTRUE(keep_julia_model)) model_var else NA_character_,
    original_nobs = n_original,
    omitted_rows = omitted_rows,
    weighted = !is.null(resolved_weights),
    fit_args = user_fit_args
  )

  model
}

.jlmer_assert_named_list <- function(x, argument) {
  if (length(x) > 0L &&
      (is.null(names(x)) || anyNA(names(x)) || any(names(x) == "") || anyDuplicated(names(x)))) {
    stop(argument, " must be a named list with unique, non-empty names.", call. = FALSE)
  }
  invisible(TRUE)
}

.jlmer_has_random_effect <- function(expr) {
  if (!is.call(expr)) {
    return(FALSE)
  }
  operator <- as.character(expr[[1L]])
  if (operator %in% c("|", "||")) {
    return(TRUE)
  }
  any(vapply(as.list(expr)[-1L], .jlmer_has_random_effect, logical(1L)))
}

.jlmer_family <- function(family, weighted = FALSE) {
  if (is.character(family)) {
    if (length(family) != 1L || is.na(family)) {
      stop("family must be a single family name, family function, or family object.", call. = FALSE)
    }
    family <- switch(
      tolower(family),
      gaussian = stats::gaussian(),
      binomial = stats::binomial(),
      poisson = stats::poisson(),
      stop(
        "Unsupported family '", family,
        "'. jlmer() supports gaussian, binomial, and poisson.",
        call. = FALSE
      )
    )
  } else if (is.function(family)) {
    family <- tryCatch(
      family(),
      error = function(e) {
        stop("Unable to construct family: ", conditionMessage(e), call. = FALSE)
      }
    )
  }

  if (!is.list(family) || is.null(family$family) || is.null(family$link)) {
    stop("family must be an R family object, family function, or supported family name.", call. = FALSE)
  }
  family_name <- tolower(as.character(family$family)[1L])
  link_name <- tolower(as.character(family$link)[1L])

  supported_links <- c(
    logit = "MixedModels.GLM.LogitLink()",
    probit = "MixedModels.GLM.ProbitLink()",
    cauchit = "MixedModels.GLM.CauchitLink()",
    log = "MixedModels.GLM.LogLink()",
    identity = "MixedModels.GLM.IdentityLink()",
    inverse = "MixedModels.GLM.InverseLink()",
    sqrt = "MixedModels.GLM.SqrtLink()",
    cloglog = "MixedModels.GLM.CloglogLink()"
  )
  if (!link_name %in% names(supported_links)) {
    stop(
      "Link '", link_name, "' cannot be converted by JellyMe4. Supported links are: ",
      paste(names(supported_links), collapse = ", "), ".",
      call. = FALSE
    )
  }

  if (identical(family_name, "gaussian")) {
    if (!identical(link_name, "identity")) {
      stop(
        "JellyMe4 can only convert a Gaussian identity-link model to lmerMod; ",
        "Gaussian link '", link_name, "' is not supported.",
        call. = FALSE
      )
    }
    julia_distribution <- "Normal()"
  } else if (identical(family_name, "binomial")) {
    julia_distribution <- if (isTRUE(weighted)) "Binomial()" else "Bernoulli()"
  } else if (identical(family_name, "poisson")) {
    julia_distribution <- "Poisson()"
  } else {
    stop(
      "Family '", family$family, "' cannot be converted by JellyMe4. ",
      "jlmer() currently supports gaussian, binomial, and poisson; ",
      "quasi and families with dispersion parameters are unsupported.",
      call. = FALSE
    )
  }

  list(
    family = family_name,
    link = link_name,
    julia_distribution = julia_distribution,
    julia_link = unname(supported_links[[link_name]])
  )
}

.jlmer_weights <- function(weights, data) {
  if (is.null(weights)) {
    return(NULL)
  }
  if (is.character(weights)) {
    if (length(weights) != 1L || is.na(weights) || !nzchar(weights)) {
      stop("A character weights argument must name exactly one data column.", call. = FALSE)
    }
    if (!weights %in% names(data)) {
      stop("Weights column not found in data: ", weights, call. = FALSE)
    }
    weights <- data[[weights]]
  }
  if (!is.numeric(weights) || length(weights) != nrow(data)) {
    stop("weights must be numeric with one value per row of data.", call. = FALSE)
  }
  if (any(is.infinite(weights)) || any(weights < 0, na.rm = TRUE)) {
    stop("weights must be non-negative and finite (or NA for na.action handling).", call. = FALSE)
  }
  as.numeric(weights)
}

.jlmer_apply_na_action <- function(model_data, na.action, original_rownames) {
  n <- nrow(model_data)
  if (is.null(na.action)) {
    return(list(keep = seq_len(n), omitted = integer(), action = NULL))
  }
  if (is.character(na.action)) {
    if (length(na.action) != 1L || is.na(na.action)) {
      stop("na.action must be a function, a single function name, or NULL.", call. = FALSE)
    }
    na.action <- tryCatch(
      match.fun(na.action),
      error = function(e) stop("Unknown na.action: ", na.action, call. = FALSE)
    )
  }
  if (!is.function(na.action)) {
    stop("na.action must be a function, a single function name, or NULL.", call. = FALSE)
  }

  rownames(model_data) <- as.character(seq_len(n))
  filtered <- tryCatch(
    na.action(model_data),
    error = function(e) {
      stop("na.action failed: ", conditionMessage(e), call. = FALSE)
    }
  )
  if (!is.data.frame(filtered)) {
    stop("na.action must return a data frame when applied to the model data.", call. = FALSE)
  }
  keep <- suppressWarnings(as.integer(rownames(filtered)))
  if (length(keep) != nrow(filtered) || anyNA(keep) || any(!keep %in% seq_len(n)) || anyDuplicated(keep)) {
    stop("na.action changed row identifiers; jlmer() cannot map the result back to data.", call. = FALSE)
  }
  omitted <- setdiff(seq_len(n), keep)
  action <- attr(filtered, "na.action")
  if (length(omitted) > 0L) {
    action_class <- if (!is.null(action) && length(class(action)) > 0L) {
      class(action)
    } else if (identical(na.action, stats::na.exclude)) {
      "exclude"
    } else {
      "omit"
    }
    action <- omitted
    names(action) <- original_rownames[omitted]
    class(action) <- action_class
  } else {
    action <- NULL
  }

  list(keep = keep, omitted = omitted, action = action)
}

.jlmer_validate_response <- function(response, family_info, weights, n_expected = NULL) {
  if (is.matrix(response) || is.data.frame(response)) {
    if (identical(family_info$family, "binomial")) {
      stop(
        "A matrix/cbind binomial response is not supported. Precompute successes / trials ",
        "and supply the trial counts with weights.",
        call. = FALSE
      )
    }
    stop("The model response must be a vector.", call. = FALSE)
  }
  if (anyNA(response)) {
    stop(
      "The evaluated response contains missing or non-finite values after na.action. ",
      "Precompute response transformations in data when necessary.",
      call. = FALSE
    )
  }
  if (length(response) == 0L) {
    stop("The model response is empty.", call. = FALSE)
  }
  if (!is.null(n_expected) && length(response) != n_expected) {
    stop("The model response must contain exactly one value per row of data.", call. = FALSE)
  }

  if (identical(family_info$family, "gaussian")) {
    if (!is.numeric(response) || any(!is.finite(response))) {
      stop("A Gaussian model requires a finite numeric response.", call. = FALSE)
    }
  } else if (identical(family_info$family, "binomial")) {
    if (is.null(weights)) {
      valid_binary <- if (is.factor(response)) {
        nlevels(droplevels(response)) == 2L
      } else if (is.logical(response)) {
        TRUE
      } else {
        is.numeric(response) && all(is.finite(response)) && all(response %in% c(0, 1))
      }
      if (!valid_binary) {
        stop(
          "An unweighted binomial model requires a 0/1, logical, or two-level factor response. ",
          "For aggregated data, use a proportion response and supply trial-count weights.",
          call. = FALSE
        )
      }
    } else {
      if (!is.numeric(response) || any(!is.finite(response)) || any(response < 0 | response > 1)) {
        stop("A weighted binomial model requires a response proportion between 0 and 1.", call. = FALSE)
      }
      if (any(!is.finite(weights)) || any(weights <= 0)) {
        stop("Binomial trial-count weights must be finite and greater than zero.", call. = FALSE)
      }
      successes <- response * weights
      if (any(abs(successes - round(successes)) > sqrt(.Machine$double.eps))) {
        stop("For a binomial model, response * weights must yield whole-number successes.", call. = FALSE)
      }
    }
  } else if (identical(family_info$family, "poisson")) {
    if (!is.numeric(response) || any(!is.finite(response)) || any(response < 0) ||
        any(abs(response - round(response)) > sqrt(.Machine$double.eps))) {
      stop("A Poisson model requires a non-negative whole-number response.", call. = FALSE)
    }
  }

  invisible(TRUE)
}
