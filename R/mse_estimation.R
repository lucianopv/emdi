# Internal documentation -------------------------------------------------------

# MSE estimation - parametric bootstrap procedure

# Function parametric_bootstrap conducts the MSE estimation defined in function
# mse_estim (see below)
# The parametric boostrap approach can be find in Molina and Rao (2010) p. 376

parametric_bootstrap <- function(framework,
                                 point_estim,
                                 fixed,
                                 transformation,
                                 interval = c(-1, 2),
                                 L,
                                 B,
                                 boot_type,
                                 parallel_mode,
                                 cpus,
                                 control,
                                 true_indicators,
                                 MSE_indicators = "all",
                                 threads = 1L) {
  message("\r", "Bootstrap started                                            ")

  # Check if C++ fast path is available
  n_standard <- 10
  use_cpp <- (boot_type == "parametric" &&
              length(framework$indicator_names) == n_standard &&
              is.null(true_indicators) &&
              cpus <= 1)

  if (use_cpp) {
    # Resolve interval defaults
    if (transformation == "box.cox" && any(interval == "default")) {
      interval <- c(-1, 2)
    } else if (transformation == "dual" && any(interval == "default")) {
      interval <- c(0, 2)
    } else if (transformation == "log.shift" && any(interval == "default")) {
      span <- range(framework$smp_data[paste(fixed[2])])
      if ((span[1] + 1) <= 1) lower <- abs(span[1]) + 1 else lower <- 0
      upper <- diff(span) / 2
      interval <- c(lower, upper)
    } else if (any(interval == "default")) {
      interval <- c(-1, 2)
    }

    # Build X_pop
    framework$pop_data[[paste0(fixed[2])]] <- seq_len(nrow(framework$pop_data))
    X_pop <- model.matrix(fixed, framework$pop_data)

    # Build X_smp
    X_smp <- model.matrix(fixed, framework$smp_data)

    # Sample domain IDs (integer factor levels)
    smp_domain_ids <- as.integer(framework$smp_domains_vec)

    # Map sample domains to population domain indices
    pop_domain_names <- as.character(unique(framework$pop_domains_vec))
    smp_domain_names <- names(table(framework$smp_domains_vec))
    smp_to_pop_map <- match(smp_domain_names, pop_domain_names)

    # Handle NULL lambda/shift
    lambda_orig <- if (is.null(point_estim$optimal_lambda)) 0 else point_estim$optimal_lambda
    shift_orig <- if (is.null(point_estim$shift_par)) 0 else point_estim$shift_par

    # Population weights
    if (!is.null(framework$pop_weights)) {
      pop_weights_vec <- as.numeric(framework$pop_data[[framework$pop_weights]])
    } else {
      pop_weights_vec <- rep(1.0, framework$N_pop)
    }

    # Aggregate domain IDs
    agg_domain_ids <- NULL
    N_dom_agg <- 0L
    if (!is.null(framework$aggregate_to_vec)) {
      agg_domain_ids <- as.integer(framework$aggregate_to_vec)
      N_dom_agg <- framework$N_dom_pop_agg
    }

    # Sample weights
    if (!is.null(framework$weights)) {
      smp_weights <- as.numeric(framework$smp_data[[framework$weights]])
    } else {
      smp_weights <- NULL
    }

    # Build indicator mask from MSE_indicators
    # Bit 0:Mean, 1:HCR, 2:PGap, 3:Gini, 4:QSR, 5:Q10, 6:Q25, 7:Q50, 8:Q75, 9:Q90
    all_ind_names <- c("Mean", "Head_Count", "Poverty_Gap", "Gini",
                       "Quintile_Share", "Quantile_10", "Quantile_25",
                       "Median", "Quantile_75", "Quantile_90")
    if (identical(MSE_indicators, "all")) {
      indicator_mask <- 0x3FFL  # all 10 indicators
    } else {
      ind_idx <- match(MSE_indicators, all_ind_names)
      if (any(is.na(ind_idx))) {
        bad <- MSE_indicators[is.na(ind_idx)]
        stop("Unknown MSE_indicators: ", paste(bad, collapse = ", "),
             ". Valid names: ", paste(all_ind_names, collapse = ", "))
      }
      indicator_mask <- sum(2^(ind_idx - 1))
    }

    # Call C++ parametric bootstrap
    mse_mat <- parametric_bootstrap_cpp(
      X_pop = X_pop,
      mu_fixed_orig = as.numeric(point_estim$gen_model$mu_fixed),
      n_pop = framework$n_pop,
      obs_dom = as.integer(framework$obs_dom),
      dist_obs_dom = as.integer(framework$dist_obs_dom),
      pop_weights = pop_weights_vec,
      N_pop = framework$N_pop,
      N_dom_pop = framework$N_dom_pop,
      X_smp = X_smp,
      n_smp = framework$n_smp,
      smp_domain_ids = smp_domain_ids,
      smp_to_pop_map = as.integer(smp_to_pop_map),
      N_smp = framework$N_smp,
      N_dom_smp = framework$N_dom_smp,
      betas_orig = as.numeric(point_estim$model_par$betas),
      sigmae2_orig = point_estim$model_par$sigmae2est,
      sigmau2_orig = point_estim$model_par$sigmau2est,
      N_dom_smp_selected = framework$N_dom_smp_selected,
      N_dom_unobs = framework$N_dom_unobs,
      B = as.integer(B),
      L = as.integer(L),
      threshold = framework$threshold,
      transformation = transformation,
      lambda_orig = lambda_orig,
      shift_orig = shift_orig,
      interval_lower = interval[1],
      interval_upper = interval[2],
      agg_domain_ids_pop = agg_domain_ids,
      N_dom_agg = N_dom_agg,
      smp_weights = smp_weights,
      indicator_mask = as.integer(indicator_mask),
      threads = as.integer(threads)
    )

    # Format result as data.frame
    if (is.null(framework$aggregate_to_vec)) {
      mses <- data.frame(Domain = unique(framework$pop_domains_vec), mse_mat)
    } else {
      mses <- data.frame(Domain = unique(framework$aggregate_to_vec), mse_mat)
    }
    colnames(mses) <- c("Domain", framework$indicator_names)

    message("\r", "Bootstrap completed", "\n")
    if (.Platform$OS.type == "windows") {
      flush.console()
    }
    return(mses)
  }

  if (boot_type == "wild") {
    res_s <- residuals(point_estim$model)
    fitted_s <- fitted(point_estim$model, level = 1)
  } else {
    res_s <- NULL
    fitted_s <- NULL
    if(is.null(control)) {
      control <- list()
    }
  }

  start_time <- Sys.time()
  if (cpus > 1) {
    cpus <- min(cpus, parallel::detectCores())
    parallelMap::parallelStart(
      mode = parallel_mode,
      cpus = cpus, show.info = FALSE
    )

    if (parallel_mode == "socket") {
      parallel::clusterSetRNGStream()
    }
    parallelMap::parallelLibrary("nlme")
    mses <- simplify2array( parallelMap::parallelLapply(
    ## mses <- parallelMap::parallelLapply(
      xs              = seq_len(B),
      fun             = mse_estim_wrapper,
      B               = B,
      framework       = framework,
      lambda          = point_estim$optimal_lambda,
      shift           = point_estim$shift_par,
      model_par       = point_estim$model_par,
      gen_model       = point_estim$gen_model,
      fixed           = fixed,
      transformation  = transformation,
      interval        = interval,
      L               = L,
      res_s           = res_s,
      fitted_s        = fitted_s,
      start_time      = start_time,
      boot_type       = boot_type,
      true_indicators = true_indicators,
      control         = control
    )
    )
    parallelMap::parallelStop()
  } else {
    mses <-  simplify2array( lapply(
    # mses <- lapply(
      X = seq_len(B),
      FUN = mse_estim_wrapper,
      B = B,
      framework = framework,
      lambda = point_estim$optimal_lambda,
      shift = point_estim$shift_par,
      model_par = point_estim$model_par,
      gen_model = point_estim$gen_model,
      fixed = fixed,
      transformation = transformation,
      interval = interval,
      L = L,
      res_s = res_s,
      fitted_s = fitted_s,
      start_time = start_time,
      boot_type = boot_type,
      true_indicators = true_indicators,
      control = control
    )
    )
  }

  message("\r", "Bootstrap completed", "\n")
  if (.Platform$OS.type == "windows") {
    flush.console()
  }

  mses <- apply(mses, c(1, 2), mean)
  if(is.null(framework$aggregate_to_vec)){
    mses <- data.frame(Domain = unique(framework$pop_domains_vec), mses)
    # mses <- list(Domain = unique(framework$pop_domains_vec), mses = mses)
  }else{
    mses <- data.frame(Domain = unique(framework$aggregate_to_vec), mses)
    # mses <- list(Domain = unique(framework$aggregate_to_vec), mses = mses)
  }

  return(mses)
}




# mse_estim (only internal) ----------------------------------------------------

# The mse_estim function defines all parameters and estimations which have to
# be replicated B times for the Parametric Bootstrap Approach.
# See Molina and Rao (2010) p. 376

mse_estim <- function(framework,
                      lambda,
                      shift,
                      model_par,
                      gen_model,
                      res_s,
                      fitted_s,
                      fixed,
                      transformation,
                      interval,
                      L,
                      boot_type,
                      control,
                      true_indicators) {



  # The function superpopulation returns an income vector and a temporary
  # variable that passes the random effect to generating bootstrap populations
  # in bootstrap_par.

  if (boot_type == "wild") {
    superpop <- superpopulation_wild(
      framework = framework,
      model_par = model_par,
      gen_model = gen_model,
      lambda = lambda,
      shift = shift,
      transformation = transformation,
      res_s = res_s,
      fitted_s = fitted_s
    )
  } else {
    superpop <- superpopulation(
      framework = framework,
      model_par = model_par,
      gen_model = gen_model,
      lambda = lambda,
      shift = shift,
      transformation = transformation
    )
  }
  pop_income_vector <- superpop$pop_income_vector

  if (inherits(framework$threshold, "function")) {
    framework$threshold <-
      framework$threshold(y = pop_income_vector)
  }

  if(!is.null(framework$aggregate_to_vec)) {
    N_dom_pop_tmp <- framework$N_dom_pop_agg
    pop_domains_vec_tmp <- framework$aggregate_to_vec
  } else {
    N_dom_pop_tmp <- framework$N_dom_pop
    pop_domains_vec_tmp <- framework$pop_domains_vec
  }

  if(!is.null(framework$pop_weights)) {
    pop_weights_vec <- framework$pop_data[[framework$pop_weights]]
  }else{
    pop_weights_vec <- rep(1, nrow(framework$pop_data))
  }

  if(is.null(true_indicators)){
    # True indicator values
    true_indicators <- matrix(
      nrow = N_dom_pop_tmp,
      data = unlist(lapply(framework$indicator_list,
                           function(f, threshold) {
                             matrix(
                               nrow = N_dom_pop_tmp,
                               data =
                                 unlist(mapply(
                                   y = split(pop_income_vector, pop_domains_vec_tmp),
                                   pop_weights = split(pop_weights_vec, pop_domains_vec_tmp),
                                   f,
                                   threshold = framework$threshold
                                 )),
                               byrow = TRUE
                             )
                           },
                           threshold = framework$threshold
                           ))
    ) } else {
      if(all(!true_indicators$Domain %in% unique(framework$pop_domains_vec))){
        stop("The domain of the true indicators does not match the domain of the framework.")
      }

      true_indicators <- as.matrix(true_indicators[,-1])

    }

  colnames(true_indicators) <- framework$indicator_names

  # The function bootstrap_par returns a sample that can be given into the
  # point estimation to get predictors of the indicators that can be compared
  # to the "truth".

  if (boot_type == "wild") {
    bootstrap_sample <- bootstrap_par_wild(
      fixed = fixed,
      transformation = transformation,
      framework = framework,
      model_par = model_par,
      lambda = lambda,
      shift = shift,
      vu_tmp = superpop$vu_tmp,
      res_s = res_s,
      fitted_s = fitted_s
    )
  } else {
    bootstrap_sample <- bootstrap_par(
      fixed = fixed,
      transformation = transformation,
      framework = framework,
      model_par = model_par,
      lambda = lambda,
      shift = shift,
      vu_tmp = superpop$vu_tmp
    )
  }

  framework$smp_data <- bootstrap_sample

  # Prediction of indicators with bootstap sample.
  bootstrap_point_estim <- as.matrix(point_estim(
    fixed = fixed,
    transformation =
      transformation,
    interval = interval,
    L = L,
    control = control,
    framework = framework
  )[[1]][, -1])

  if(ncol(true_indicators) != ncol(bootstrap_point_estim)){
        stop("The number of indicators in the true indicators does not match the number of indicators in the framework.")
  }


  ## return(list(bootstrap_point_estim = bootstrap_point_estim,
  ##             true_indicators = true_indicators,
  ##             superpop = superpop))

  return((bootstrap_point_estim - true_indicators)^2)
} # End mse_estim


# Superpopulation function -----------------------------------------------------

# The model parameter from the nested error linear regression model are
# used to contruct a superpopulation model.
superpopulation_wild <- function(framework, model_par, gen_model, lambda,
                                 shift, transformation, res_s, fitted_s) {
  # rescaling the errors
  res_s <- sqrt(model_par$sigmae2est) * (res_s - mean(res_s)) / sd(res_s)

  # superpopulation random effect
  vu_tmp <- rnorm(framework$N_dom_pop, 0, sqrt(model_par$sigmau2est))
  vu_pop <- rep(vu_tmp, framework$n_pop)

  # income without individual errors
  Y_pop_b <- gen_model$mu_fixed + vu_pop

  indexer <- vapply(Y_pop_b,
    function(x) {
      which.min(abs(x - fitted_s))
    },
    FUN.VALUE = integer(1)
  )

  # superpopulation individual errors
  eps <- res_s[indexer]
  wu <- sample(c(-1, 1), size = length(eps), replace = TRUE)
  eps <- abs(eps) * wu

  #  superpopulation income vector
  Y_pop_b <- Y_pop_b + eps

  Y_pop_b <- back_transformation(
    y = Y_pop_b,
    transformation = transformation,
    lambda = lambda,
    shift = shift
  )
  Y_pop_b[!is.finite(Y_pop_b)] <- 0

  return(list(pop_income_vector = Y_pop_b, vu_tmp = vu_tmp))
}

superpopulation <- function(framework, model_par, gen_model, lambda, shift,
                            transformation) {
  lambda_val <- if (is.null(lambda)) 0 else lambda
  shift_val <- if (is.null(shift)) 0 else shift

  result <- gen_superpop_cpp(
    mu_fixed = as.numeric(gen_model$mu_fixed),
    sigmae2 = model_par$sigmae2est,
    sigmau2 = model_par$sigmau2est,
    obs_dom = as.integer(framework$obs_dom),
    n_pop = framework$n_pop,
    N_dom_pop = framework$N_dom_pop,
    transformation = transformation,
    lambda = lambda_val,
    shift = shift_val
  )

  return(list(
    pop_income_vector = as.numeric(result$pop_income_vector),
    vu_tmp = as.numeric(result$vu_tmp),
    eps = as.numeric(result$eps),
    vu_pop = as.numeric(result$vu_pop),
    Y_pop_b_notrans = as.numeric(result$Y_pop_b_notrans)
  ))
}

# Bootstrap function -----------------------------------------------------------

bootstrap_par <- function(fixed,
                          transformation,
                          framework,
                          model_par,
                          lambda,
                          shift,
                          vu_tmp) {
  lambda_val <- if (is.null(lambda)) 0 else lambda
  shift_val <- if (is.null(shift)) 0 else shift

  X_smp <- model.matrix(fixed, framework$smp_data)

  # Map sample domains to population domain indices
  pop_domain_names <- as.character(unique(framework$pop_domains_vec))
  smp_domain_names <- names(table(framework$smp_domains_vec))
  smp_to_pop_map <- match(smp_domain_names, pop_domain_names)

  Y_smp_b <- gen_bootstrap_sample_cpp(
    X_smp = X_smp,
    betas = as.numeric(model_par$betas),
    sigmae2 = model_par$sigmae2est,
    sigmau2 = model_par$sigmau2est,
    vu_tmp = vu_tmp,
    smp_to_pop_map = as.integer(smp_to_pop_map),
    n_smp = framework$n_smp,
    transformation = transformation,
    lambda = lambda_val,
    shift = shift_val
  )

  bootstrap_smp <- framework$smp_data
  bootstrap_smp[paste(fixed[2])] <- as.numeric(Y_smp_b)

  return(bootstrap_sample = bootstrap_smp)
}

bootstrap_par_wild <- function(fixed,
                               transformation,
                               framework,
                               model_par,
                               lambda,
                               shift,
                               vu_tmp,
                               res_s,
                               fitted_s) {
  # rescaling sample individual error term
  res_s <- sqrt(model_par$sigmae2est) * (res_s - mean(res_s)) / sd(res_s)
  # Bootstrap sample individual error term
  ws <- sample(c(-1, 1), size = length(res_s), replace = TRUE)
  eps <- abs(res_s) * ws

  # Bootstrap sample random effect
  # Match random effects by domain name to handle selected_domains
  pop_domain_names <- as.character(unique(framework$pop_domains_vec))
  smp_domain_names <- names(table(framework$smp_domains_vec))
  
  # Create a vector to hold random effects for all sample domains
  # When selected_domains is used, some sample domains may not be in the
  # selected set. For these domains, we still need to generate bootstrap
  # samples (since the model uses all sample data), so we generate new
  # random effects from the estimated distribution.
  vu_for_smp <- numeric(length(smp_domain_names))
  for (i in seq_along(smp_domain_names)) {
    # Find this sample domain in the population domains
    pop_idx <- which(pop_domain_names == smp_domain_names[i])
    if (length(pop_idx) > 0) {
      # This domain is in the selected population domains
      vu_for_smp[i] <- vu_tmp[pop_idx]
    } else {
      # This domain is not in the selected set, generate new random effect
      vu_for_smp[i] <- rnorm(1, 0, sqrt(model_par$sigmau2est))
    }
  }
  
  vu_smp <- rep(vu_for_smp, framework$n_smp)

  # Extraction of design matrix
  X_smp <- model.matrix(fixed, framework$smp_data)

  # Transformed bootstrap income vector
  Y_smp_b <- X_smp %*% model_par$betas + eps + vu_smp

  # Back transformation of bootstrap income vector
  Y_smp_b <- back_transformation(
    y = Y_smp_b,
    transformation = transformation,
    lambda = lambda,
    shift = shift
  )
  Y_smp_b[!is.finite(Y_smp_b)] <- 0

  # Inclusion of bootstrap income vector into sample data
  bootstrap_smp <- framework$smp_data
  bootstrap_smp[paste(fixed[2])] <- Y_smp_b

  return(bootstrap_sample = bootstrap_smp)
}

# progress for mse_estim (only internal) ----------

mse_estim_wrapper <- function(i,
                              B,
                              framework,
                              lambda,
                              shift,
                              model_par,
                              gen_model,
                              fixed,
                              transformation,
                              interval,
                              L,
                              res_s,
                              fitted_s,
                              start_time,
                              boot_type,
                              true_indicators,
                              control) {
  tmp <- mse_estim(
    framework = framework,
    lambda = lambda,
    shift = shift,
    model_par = model_par,
    gen_model = gen_model,
    res_s = res_s,
    fitted_s = fitted_s,
    fixed = fixed,
    transformation = transformation,
    interval = interval,
    L = L,
    boot_type = boot_type,
    control = control,
    true_indicators = true_indicators
  )

  # Progress. This runs per iteration, possibly inside a parallelMap worker, so
  # it cannot hold state across calls -- hence the stateless iteration-count
  # throttle rather than the time-based one progress_reporter() uses. The line
  # itself comes from the shared formatter so all progress output in the package
  # reads the same.
  #
  # The previous version built its message across two source lines, which
  # embedded a literal tab and newline and so defeated its own leading "\r".
  if (i %% 10 == 0 && i != B) {
    elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    message("\r", progress_line(i, B, elapsed, start_time, "bootstrap iteration"),
            appendLF = FALSE)
    if (.Platform$OS.type == "windows") flush.console()
  }
  return(tmp)
}
