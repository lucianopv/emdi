# Internal documentation -------------------------------------------------------

# Point estimation function

# This function implements the transformation of data, estimation of the nested
# error linear regression model and the monte-carlo approximation to predict
# the desired indicators. If the weighted version of the approach is used, then
# additional estimation steps are taken in order to calculate weighted
# regression coefficients before the monte-carlo approximation. See
# corresponding functions below.


point_estim <- function(framework,
                        fixed,
                        transformation,
                        interval,
                        L,
                        keep_data = FALSE,
                        control = list()) {

  # Transformation of data -----------------------------------------------------

  # Estimating the optimal parameter by optimization
  # Optimal parameter function returns the minimum of the optimization
  # functions from generic_opt; the minimum is the optimal lambda.
  # The function can be found in the script optimal_parameter.R
  optimal_lambda <- optimal_parameter(
    generic_opt = generic_opt,
    fixed = fixed,
    smp_data = framework$smp_data,
    smp_domains = framework$smp_domains,
    transformation = transformation,
    interval = interval,
    control = control
  )

  # Data_transformation function returns transformed data and shift parameter.
  # The function can be found in the script transformation_functions.R
  transformation_par <- data_transformation(
    fixed = fixed,
    smp_data = framework$smp_data,
    transformation = transformation,
    lambda = optimal_lambda
  )
  shift_par <- transformation_par$shift

  # Model estimation, model parameter and parameter of generating model --------

  # Estimation of the nested error linear regression model
  # See Molina and Rao (2010) p. 374
  # lme function is included in the nlme package which is imported.

  mixed_model <- nlme::lme(
    fixed = fixed,
    data = transformation_par$transformed_data,
    random =
      as.formula(paste0(
        "~ 1 | as.factor(",
        framework$smp_domains, ")"
      )),
    method = "REML",
    keep.data = keep_data,
    control = control
  )

  # Function model_par extracts the needed parameters theta from the nested
  # error linear regression model. It returns the beta coefficients (betas),
  # sigmae2est, sigmau2est and the random effect (rand_eff).

  est_par <- model_par(
    mixed_model = mixed_model,
    framework = framework,
    fixed = fixed,
    transformation_par = transformation_par
  )

  # Function gen_model calculates the parameters in the generating model.
  # See Molina and Rao (2010) p. 375 (20)
  # The function returns sigmav2est and the constant part mu.
  gen_par <- gen_model(
    model_par = est_par,
    fixed = fixed,
    framework = framework
  )

  # Monte-Carlo approximation --------------------------------------------------
  if (inherits(framework$threshold, "function")) {
    framework$threshold <-
      framework$threshold(
        y =
          as.numeric(framework$smp_data[[paste0(fixed[2])]])
      )
  }

  # The monte-carlo function returns a data frame of desired indicators.
  indicator_prediction <- monte_carlo(
    transformation = transformation,
    L = L,
    framework = framework,
    lambda = optimal_lambda,
    shift = shift_par,
    model_par = est_par,
    gen_model = gen_par
  )

  mixed_model$coefficients_weighted <- if (!is.null(framework$weights)) {
    as.numeric(est_par$betas)
  } else {
    NULL
  }
  names(mixed_model$coefficients_weighted) <- if (!is.null(framework$weights)) {
    rownames(est_par$betas)
  } else {
    NULL
  }
  return(list(
    ind = indicator_prediction$point_estimates,
    y_mcmc = indicator_prediction$y_mcmc,
    optimal_lambda = optimal_lambda,
    shift_par = shift_par,
    model_par = est_par,
    gen_model = gen_par,
    model = mixed_model
  ))
} # End point estimation function


# All following functions are only internal ------------------------------------

# Functions to extract and calculate model parameter----------------------------

# Function model_par extracts the needed parameters theta from the nested
# error linear regression model. It returns the beta coefficients (betas),
# sigmae2est, sigmau2est and the random effect (rand_eff).

model_par <- function(framework,
                      mixed_model,
                      fixed,
                      transformation_par) {
  if (is.null(framework$weights)) {
    # fixed parametersn
    betas <- nlme::fixed.effects(mixed_model)
    # Estimated error variance
    sigmae2est <- mixed_model$sigma^2
    # VarCorr(fit2) is the estimated random error variance
    sigmau2est <- as.numeric(nlme::VarCorr(mixed_model)[1, 1])
    # Random effect: vector with zeros for all domains, filled with
    rand_eff <- rep(0, length(unique(framework$pop_domains_vec)))
    # random effect for in-sample domains (dist_obs_dom)
    # Extract random effects and match by domain name
    rand_effects_all <- random.effects(mixed_model)[[1]]
    smp_domain_names <- rownames(rand_effects_all)
    pop_domain_names <- as.character(unique(framework$pop_domains_vec))
    
    # For each population domain that is in sample, get its random effect
    for (i in seq_along(pop_domain_names)) {
      if (framework$dist_obs_dom[i]) {
        # Find the position of this domain in the sample domains
        smp_idx <- which(smp_domain_names == pop_domain_names[i])
        if (length(smp_idx) > 0) {
          rand_eff[i] <- rand_effects_all[smp_idx]
        }
      }
    }

    return(list(
      betas = betas,
      sigmae2est = sigmae2est,
      sigmau2est = sigmau2est,
      rand_eff = rand_eff
    ))
  } else {
    # fixed parameters
    betas <- nlme::fixed.effects(mixed_model)
    # Estimated error variance
    sigmae2est <- mixed_model$sigma^2
    # VarCorr(fit2) is the estimated random error variance
    sigmau2est <- as.numeric(nlme::VarCorr(mixed_model)[1, 1])

    # Calculations needed for pseudo EB

    weight_sum <- rep(0, framework$N_dom_smp)
    mean_dep <- rep(0, framework$N_dom_smp)
    mean_indep <- matrix(0, nrow = framework$N_dom_smp, ncol = length(betas))
    delta2 <- rep(0, framework$N_dom_smp)
    gamma_weight <- rep(0, framework$N_dom_smp)
    num <- matrix(0, nrow = length(betas), ncol = 1)
    den <- matrix(0, nrow = length(betas), ncol = length(betas))

    for (d in 1:framework$N_dom_smp) {
      domain <- names(table(framework$smp_domains_vec)[d])

      # Domain means of of the dependent variable
      dep_smp <- transformation_par$transformed_data[[
      as.character(mixed_model$terms[[2]])]][
        framework$smp_domains_vec == domain
      ]
      weight_smp <- transformation_par$transformed_data[[
      as.character(framework$weights)]][framework$smp_domains_vec == domain]
      weight_sum[d] <- sum(weight_smp)

      indep_smp <- if(length(weight_smp) == 1) {
        matrix(model.matrix(fixed, framework$smp_data)[framework$smp_domains_vec == domain,]
               , ncol = length(betas), nrow = 1)
      } else {
        model.matrix(fixed, framework$smp_data)[framework$smp_domains_vec == domain,]
      }

      # weighted mean of the dependent variable
      mean_dep[d] <- sum(weight_smp * dep_smp) / weight_sum[d]

      # weighted means of the auxiliary information
      for (k in 1:length(betas)) {
        mean_indep[d, k] <- sum(weight_smp * indep_smp[, k]) / weight_sum[d]
      }

      delta2[d] <- sum(weight_smp^2) / (weight_sum[d]^2)
      gamma_weight[d] <- sigmau2est / (sigmau2est + sigmae2est * delta2[d])
      weight_smp_diag <- diag(weight_smp)
      dep_var_ast <- dep_smp - gamma_weight[d] * mean_dep[d]
      indep_weight <- t(indep_smp) %*% weight_smp_diag
      indep_var_ast <- indep_smp - matrix(rep(
        gamma_weight[d] *
          mean_indep[d, ],
        framework$n_smp[d]
      ),
      nrow = framework$n_smp[d],
      byrow = TRUE
      )



      num <- num + (indep_weight %*% dep_var_ast)
      den <- den + (indep_weight %*% indep_var_ast)
    }


    betas <- solve(den) %*% num
    # Random effect: vector with zeros for all domains, filled with
    rand_eff <- rep(0, length(unique(framework$pop_domains_vec)))
    
    # Map random effects to population domains by matching domain names
    smp_domain_names <- names(table(framework$smp_domains_vec))
    pop_domain_names <- as.character(unique(framework$pop_domains_vec))
    
    for (i in seq_along(pop_domain_names)) {
      if (framework$dist_obs_dom[i]) {
        # Find the position of this domain in the sample domains
        smp_idx <- which(smp_domain_names == pop_domain_names[i])
        if (length(smp_idx) > 0) {
          rand_eff[i] <- gamma_weight[smp_idx] * (mean_dep[smp_idx] -
            mean_indep[smp_idx, ] %*% betas)
        }
      }
    }


    return(list(
      betas = betas,
      sigmae2est = sigmae2est,
      sigmau2est = sigmau2est,
      rand_eff = rand_eff,
      gammaw = gamma_weight,
      delta2 = delta2
    ))
  }
} # End model_par



# Function gen_model calculates the parameters in the generating model.
# See Molina and Rao (2010) p. 375 (20)
gen_model <- function(fixed,
                      framework,
                      model_par) {
  if (is.null(framework$weights)) {
    # Parameter for calculating variance of new random effect
    gamma <- model_par$sigmau2est / (model_par$sigmau2est +
      model_par$sigmae2est / framework$n_smp)
    # Variance of new random effect
    sigmav2est_all <- model_par$sigmau2est * (1 - gamma)
    
    # Extract sigmav2est only for selected domains that are in sample
    # Match by domain name to handle selected_domains filtering
    smp_domain_names <- names(table(framework$smp_domains_vec))
    pop_domain_names <- as.character(unique(framework$pop_domains_vec))
    pop_domain_in_smp <- pop_domain_names[framework$dist_obs_dom]
    
    # Find indices of selected domains in sample domain list
    sigmav2est_indices <- match(pop_domain_in_smp, smp_domain_names)
    sigmav2est <- sigmav2est_all[sigmav2est_indices]
    
    # Random effect in constant part of y for in-sample households
    rand_eff_pop <- rep(model_par$rand_eff, framework$n_pop)
    # Model matrix for population covariate information
    framework$pop_data[[paste0(fixed[2])]] <- seq_len(nrow(framework$pop_data))
    X_pop <- model.matrix(fixed, framework$pop_data)

    # Constant part of predicted y
    mu_fixed <- X_pop %*% model_par$betas
    mu <- mu_fixed + rand_eff_pop

    return(list(sigmav2est = sigmav2est, mu = mu, mu_fixed = mu_fixed))
  } else {
    # Parameter for calculating variance of new random effect
    gamma <- model_par$gammaw
    # Variance of new random effect
    sigmav2est_all <- model_par$sigmau2est * (1 - gamma)
    
    # Extract sigmav2est only for selected domains that are in sample
    # Match by domain name to handle selected_domains filtering
    smp_domain_names <- names(table(framework$smp_domains_vec))
    pop_domain_names <- as.character(unique(framework$pop_domains_vec))
    pop_domain_in_smp <- pop_domain_names[framework$dist_obs_dom]
    
    # Find indices of selected domains in sample domain list
    sigmav2est_indices <- match(pop_domain_in_smp, smp_domain_names)
    sigmav2est <- sigmav2est_all[sigmav2est_indices]
    
    # Random effect in constant part of y for in-sample households
    rand_eff_pop <- rep(model_par$rand_eff, framework$n_pop) ####### change
    # Model matrix for population covariate information
    framework$pop_data[[paste0(fixed[2])]] <- seq_len(nrow(framework$pop_data))
    X_pop <- model.matrix(fixed, framework$pop_data)

    # Constant part of predicted y
    mu_fixed <- X_pop %*% model_par$betas
    mu <- mu_fixed + rand_eff_pop


    return(list(sigmav2est = sigmav2est, mu = mu, mu_fixed = mu_fixed))
  }
} # End gen_model


# Monte-Carlo approximation ----------------------------------------------------

# The function approximates the expected value (Molina and Rao (2010)
# p.372 (6)). For description of monte-carlo simulation see Molina and
# Rao (2010) p. 373 (13) and p. 374-375
monte_carlo <- function(transformation,
                        L,
                        framework,
                        lambda = NULL,
                        shift = NULL,
                        model_par,
                        gen_model) {

  # Handle aggregate_to
  if(!is.null(framework$aggregate_to_vec)){
    N_dom_pop_tmp <- framework$N_dom_pop_agg
    pop_domains_vec_tmp <- framework$aggregate_to_vec
  } else {
    N_dom_pop_tmp <- framework$N_dom_pop
    pop_domains_vec_tmp <- framework$pop_domains_vec
  }

  # Population weights
  if(!is.null(framework$pop_weights)){
    pop_weights_vec <- framework$pop_data[[framework$pop_weights]]
  } else {
    pop_weights_vec <- rep(1, nrow(framework$pop_data))
  }

  # Ensure lambda/shift are numeric (not NULL)
  lambda_val <- if (is.null(lambda)) 0 else lambda
  shift_val <- if (is.null(shift)) 0 else shift

  # Prepare optional aggregate domain IDs for C++
  agg_ids <- NULL
  N_dom_agg <- 0L
  if (!is.null(framework$aggregate_to_vec)) {
    agg_ids <- as.integer(pop_domains_vec_tmp)
    N_dom_agg <- N_dom_pop_tmp
  }

  # Call C++ Monte Carlo implementation
  result_cpp <- monte_carlo_cpp(
    mu = as.numeric(gen_model$mu),
    sigmae2 = model_par$sigmae2est,
    sigmau2 = model_par$sigmau2est,
    sigmav2 = gen_model$sigmav2est,
    domain_ids = as.integer(framework$pop_domains_vec),
    obs_dom = as.integer(framework$obs_dom),
    dist_obs_dom = as.integer(framework$dist_obs_dom),
    n_pop = framework$n_pop,
    N_dom_pop = framework$N_dom_pop,
    N_dom_smp = framework$N_dom_smp_selected,
    N_dom_unobs = framework$N_dom_unobs,
    L = as.integer(L),
    threshold = framework$threshold,
    transformation = transformation,
    lambda = lambda_val,
    shift = shift_val,
    pop_weights = pop_weights_vec,
    n_indicators = 10L,
    agg_domain_ids = agg_ids,
    N_dom_agg = N_dom_agg
  )

  if (!is.null(framework$aggregate_to_vec)) {
    # C++ already computed indicators on aggregated domains
    n_std <- 10
    if (length(framework$indicator_names) > n_std) {
      # Custom indicators: compute via R using y_mcmc from C++
      custom_list <- framework$indicator_list[(n_std + 1):length(framework$indicator_list)]
      n_custom <- length(framework$indicator_names) - n_std
      custom_ests <- array(dim = c(N_dom_pop_tmp, L, n_custom))
      for (l in seq_len(L)) {
        custom_ests[, l, ] <-
          matrix(
            nrow = N_dom_pop_tmp,
            data = unlist(lapply(custom_list,
              function(f, threshold) {
                matrix(
                  nrow = N_dom_pop_tmp,
                  data = unlist(mapply(
                    y = split(result_cpp$y_mcmc[, l], pop_domains_vec_tmp),
                    pop_weights = split(pop_weights_vec, pop_domains_vec_tmp),
                    f,
                    threshold = framework$threshold
                  )), byrow = TRUE
                )
              },
              threshold = framework$threshold
            ))
          )
      }
      custom_means <- apply(custom_ests, c(3), rowMeans)
      if (!is.matrix(custom_means)) custom_means <- matrix(custom_means, ncol = n_custom)
      point_estimates <- data.frame(
        Domain = unique(pop_domains_vec_tmp),
        result_cpp$point_estimates[, 1:n_std],
        custom_means
      )
    } else {
      point_estimates <- data.frame(
        Domain = unique(pop_domains_vec_tmp),
        result_cpp$point_estimates[, seq_along(framework$indicator_names)]
      )
    }
  } else {
    # Standard case: C++ computed standard 10 indicators
    n_std <- 10
    if (length(framework$indicator_names) > n_std) {
      # Custom indicators: compute via R using y_mcmc from C++
      custom_list <- framework$indicator_list[(n_std + 1):length(framework$indicator_list)]
      n_custom <- length(framework$indicator_names) - n_std
      custom_ests <- array(dim = c(N_dom_pop_tmp, L, n_custom))
      for (l in seq_len(L)) {
        custom_ests[, l, ] <-
          matrix(
            nrow = N_dom_pop_tmp,
            data = unlist(lapply(custom_list,
              function(f, threshold) {
                matrix(
                  nrow = N_dom_pop_tmp,
                  data = unlist(mapply(
                    y = split(result_cpp$y_mcmc[, l], pop_domains_vec_tmp),
                    pop_weights = split(pop_weights_vec, pop_domains_vec_tmp),
                    f,
                    threshold = framework$threshold
                  )), byrow = TRUE
                )
              },
              threshold = framework$threshold
            ))
          )
      }
      custom_means <- apply(custom_ests, c(3), rowMeans)
      if (!is.matrix(custom_means)) custom_means <- matrix(custom_means, ncol = n_custom)
      point_estimates <- data.frame(
        Domain = unique(pop_domains_vec_tmp),
        result_cpp$point_estimates[, 1:n_std],
        custom_means
      )
    } else {
      point_estimates <- data.frame(
        Domain = unique(pop_domains_vec_tmp),
        result_cpp$point_estimates[, seq_along(framework$indicator_names)]
      )
    }
  }
  colnames(point_estimates) <- c("Domain", framework$indicator_names)

  return(list("point_estimates" = point_estimates,
              "y_mcmc" = result_cpp$y_mcmc))
} # End Monte-Carlo


# The function errors_gen returns error terms of the generating model.
# See Molina and Rao (2010) p. 375 (20)

errors_gen <- function(framework, model_par, gen_model) {
  # individual error term in generating model epsilon
  epsilon <- rnorm(framework$N_pop, 0, sqrt(model_par$sigmae2est))

  # empty vector for new random effect in generating model
  vu <- vector(length = framework$N_pop)
  # new random effect for out-of-sample domains
  vu[!framework$obs_dom] <- rep(
    rnorm(
      framework$N_dom_unobs,
      0,
      sqrt(model_par$sigmau2est)
    ),
    framework$n_pop[!framework$dist_obs_dom]
  )
  # new random effect for in-sample-domains
  vu[framework$obs_dom] <- rep(
    rnorm(
      rep(1, framework$N_dom_smp_selected),
      0,
      sqrt(gen_model$sigmav2est)
    ),
    framework$n_pop[framework$dist_obs_dom]
  )

  return(list(epsilon = epsilon, vu = vu))
} # End errors_gen

# The function prediction_y returns a predicted income vector which can be used
# to calculate indicators. Note that a whole income vector is predicted without
# distinction between in- and out-of-sample domains.
prediction_y <- function(transformation,
                         lambda,
                         shift,
                         gen_model,
                         errors_gen,
                         framework) {

  # predicted population income vector
  y_pred <- gen_model$mu + errors_gen$epsilon + errors_gen$vu

  # back-transformation of predicted population income vector
  y_pred <- back_transformation(
    y = y_pred,
    transformation = transformation,
    lambda = lambda,
    shift = shift
  )
  y_pred[!is.finite(y_pred)] <- 0

  return(y_pred)
} # End prediction_y
