# Tests for parametric_bootstrap_cpp

test_that("parametric_bootstrap_cpp runs without error (direct call, log)", {
  data("eusilcA_smp", package = "emdi2")
  data("eusilcA_pop", package = "emdi2")

  fixed <- eqIncome ~ gender + eqsize
  framework <- framework_ebp(fixed = fixed,
    pop_data = eusilcA_pop, pop_domains = "district",
    smp_data = eusilcA_smp, smp_domains = "district",
    threshold = 10924.32, custom_indicator = NULL,
    na.rm = TRUE, pop_weights = NULL, weights = NULL)

  set.seed(42)
  pe <- point_estim(framework = framework, fixed = fixed,
    transformation = "log", interval = "default", L = 5)

  # Prepare inputs
  framework$pop_data$eqIncome <- seq_len(nrow(framework$pop_data))
  X_pop <- model.matrix(fixed, framework$pop_data)

  pop_domain_names <- as.character(unique(framework$pop_domains_vec))
  smp_domain_names <- names(table(framework$smp_domains_vec))
  smp_to_pop_map <- match(smp_domain_names, pop_domain_names)

  X_smp <- model.matrix(fixed, framework$smp_data)
  smp_domain_ids <- as.integer(as.factor(framework$smp_data$district))

  lambda_orig <- if (is.null(pe$optimal_lambda)) 0 else pe$optimal_lambda
  shift_orig <- if (is.null(pe$shift_par)) 0 else pe$shift_par

  set.seed(42)
  mse <- parametric_bootstrap_cpp(
    X_pop = X_pop,
    mu_fixed_orig = as.numeric(pe$gen_model$mu_fixed),
    n_pop = framework$n_pop,
    obs_dom = as.integer(framework$obs_dom),
    dist_obs_dom = as.integer(framework$dist_obs_dom),
    pop_weights = rep(1.0, framework$N_pop),
    N_pop = framework$N_pop,
    N_dom_pop = framework$N_dom_pop,
    X_smp = X_smp,
    n_smp = framework$n_smp,
    smp_domain_ids = smp_domain_ids,
    smp_to_pop_map = as.integer(smp_to_pop_map),
    N_smp = framework$N_smp,
    N_dom_smp = framework$N_dom_smp,
    betas_orig = as.numeric(pe$model_par$betas),
    sigmae2_orig = pe$model_par$sigmae2est,
    sigmau2_orig = pe$model_par$sigmau2est,
    N_dom_smp_selected = framework$N_dom_smp_selected,
    N_dom_unobs = framework$N_dom_unobs,
    B = 3L, L = 5L,
    threshold = 10924.32,
    transformation = "log",
    lambda_orig = lambda_orig,
    shift_orig = shift_orig,
    interval_lower = -1, interval_upper = 2
  )

  expect_equal(nrow(mse), framework$N_dom_pop)
  expect_equal(ncol(mse), 10)
  expect_true(all(mse >= 0))
})

test_that("parametric_bootstrap_cpp runs without error (direct call, box.cox)", {
  data("eusilcA_smp", package = "emdi2")
  data("eusilcA_pop", package = "emdi2")

  fixed <- eqIncome ~ gender + eqsize
  framework <- framework_ebp(fixed = fixed,
    pop_data = eusilcA_pop, pop_domains = "district",
    smp_data = eusilcA_smp, smp_domains = "district",
    threshold = 10924.32, custom_indicator = NULL,
    na.rm = TRUE, pop_weights = NULL, weights = NULL)

  set.seed(42)
  pe <- point_estim(framework = framework, fixed = fixed,
    transformation = "box.cox", interval = "default", L = 5)

  # Prepare inputs
  framework$pop_data$eqIncome <- seq_len(nrow(framework$pop_data))
  X_pop <- model.matrix(fixed, framework$pop_data)

  pop_domain_names <- as.character(unique(framework$pop_domains_vec))
  smp_domain_names <- names(table(framework$smp_domains_vec))
  smp_to_pop_map <- match(smp_domain_names, pop_domain_names)

  X_smp <- model.matrix(fixed, framework$smp_data)
  smp_domain_ids <- as.integer(as.factor(framework$smp_data$district))

  lambda_orig <- if (is.null(pe$optimal_lambda)) 0 else pe$optimal_lambda
  shift_orig <- if (is.null(pe$shift_par)) 0 else pe$shift_par

  set.seed(42)
  mse <- parametric_bootstrap_cpp(
    X_pop = X_pop,
    mu_fixed_orig = as.numeric(pe$gen_model$mu_fixed),
    n_pop = framework$n_pop,
    obs_dom = as.integer(framework$obs_dom),
    dist_obs_dom = as.integer(framework$dist_obs_dom),
    pop_weights = rep(1.0, framework$N_pop),
    N_pop = framework$N_pop,
    N_dom_pop = framework$N_dom_pop,
    X_smp = X_smp,
    n_smp = framework$n_smp,
    smp_domain_ids = smp_domain_ids,
    smp_to_pop_map = as.integer(smp_to_pop_map),
    N_smp = framework$N_smp,
    N_dom_smp = framework$N_dom_smp,
    betas_orig = as.numeric(pe$model_par$betas),
    sigmae2_orig = pe$model_par$sigmae2est,
    sigmau2_orig = pe$model_par$sigmau2est,
    N_dom_smp_selected = framework$N_dom_smp_selected,
    N_dom_unobs = framework$N_dom_unobs,
    B = 3L, L = 5L,
    threshold = 10924.32,
    transformation = "box.cox",
    lambda_orig = lambda_orig,
    shift_orig = shift_orig,
    interval_lower = -1, interval_upper = 2
  )

  expect_equal(nrow(mse), framework$N_dom_pop)
  expect_equal(ncol(mse), 10)
  expect_true(all(mse >= 0))
})

test_that("parametric_bootstrap_cpp works with aggregate domains (direct call)", {
  data("eusilcA_smp", package = "emdi2")
  data("eusilcA_pop", package = "emdi2")

  eusilcA_pop$region <- as.integer(as.factor(eusilcA_pop$district)) %% 5 + 1

  fixed <- eqIncome ~ gender + eqsize
  framework <- framework_ebp(fixed = fixed,
    pop_data = eusilcA_pop, pop_domains = "district",
    smp_data = eusilcA_smp, smp_domains = "district",
    threshold = 10924.32, custom_indicator = NULL,
    na.rm = TRUE, pop_weights = NULL, weights = NULL,
    aggregate_to = "region")

  set.seed(42)
  pe <- point_estim(framework = framework, fixed = fixed,
    transformation = "log", interval = "default", L = 5)

  # Prepare inputs
  framework$pop_data$eqIncome <- seq_len(nrow(framework$pop_data))
  X_pop <- model.matrix(fixed, framework$pop_data)

  pop_domain_names <- as.character(unique(framework$pop_domains_vec))
  smp_domain_names <- names(table(framework$smp_domains_vec))
  smp_to_pop_map <- match(smp_domain_names, pop_domain_names)

  X_smp <- model.matrix(fixed, framework$smp_data)
  smp_domain_ids <- as.integer(as.factor(framework$smp_data$district))

  lambda_orig <- if (is.null(pe$optimal_lambda)) 0 else pe$optimal_lambda
  shift_orig <- if (is.null(pe$shift_par)) 0 else pe$shift_par

  # Prepare aggregate domain IDs
  agg_ids <- as.integer(framework$aggregate_to_vec)
  N_dom_agg <- framework$N_dom_pop_agg

  set.seed(42)
  mse <- parametric_bootstrap_cpp(
    X_pop = X_pop,
    mu_fixed_orig = as.numeric(pe$gen_model$mu_fixed),
    n_pop = framework$n_pop,
    obs_dom = as.integer(framework$obs_dom),
    dist_obs_dom = as.integer(framework$dist_obs_dom),
    pop_weights = rep(1.0, framework$N_pop),
    N_pop = framework$N_pop,
    N_dom_pop = framework$N_dom_pop,
    X_smp = X_smp,
    n_smp = framework$n_smp,
    smp_domain_ids = smp_domain_ids,
    smp_to_pop_map = as.integer(smp_to_pop_map),
    N_smp = framework$N_smp,
    N_dom_smp = framework$N_dom_smp,
    betas_orig = as.numeric(pe$model_par$betas),
    sigmae2_orig = pe$model_par$sigmae2est,
    sigmau2_orig = pe$model_par$sigmau2est,
    N_dom_smp_selected = framework$N_dom_smp_selected,
    N_dom_unobs = framework$N_dom_unobs,
    B = 3L, L = 5L,
    threshold = 10924.32,
    transformation = "log",
    lambda_orig = lambda_orig,
    shift_orig = shift_orig,
    interval_lower = -1, interval_upper = 2,
    agg_domain_ids_pop = agg_ids,
    N_dom_agg = N_dom_agg
  )

  expect_equal(nrow(mse), N_dom_agg)
  expect_equal(ncol(mse), 10)
  expect_true(all(mse >= 0))
})

# Tests that will pass after Task 5 wires R function to call C++
test_that("parametric_bootstrap_cpp produces valid MSE via ebp (log transformation)", {
  skip("Requires Task 5: wiring R parametric_bootstrap to call C++")
  data("eusilcA_smp", package = "emdi2")
  data("eusilcA_pop", package = "emdi2")

  set.seed(42)
  result <- ebp(
    fixed = eqIncome ~ gender + eqsize,
    pop_data = eusilcA_pop, pop_domains = "district",
    smp_data = eusilcA_smp, smp_domains = "district",
    L = 5, MSE = TRUE, B = 3, transformation = "log"
  )

  mse <- estimators(result, indicator = "all", MSE = TRUE)
  mse_cols <- grep("_MSE$", names(mse$ind), value = TRUE)
  for (col in mse_cols) {
    expect_true(all(mse$ind[[col]] >= 0), info = paste("MSE:", col))
  }
})

test_that("parametric_bootstrap_cpp produces valid MSE via ebp (box.cox transformation)", {
  skip("Requires Task 5: wiring R parametric_bootstrap to call C++")
  data("eusilcA_smp", package = "emdi2")
  data("eusilcA_pop", package = "emdi2")

  set.seed(42)
  result <- ebp(
    fixed = eqIncome ~ gender + eqsize,
    pop_data = eusilcA_pop, pop_domains = "district",
    smp_data = eusilcA_smp, smp_domains = "district",
    L = 5, MSE = TRUE, B = 3, transformation = "box.cox"
  )

  mse <- estimators(result, indicator = "all", MSE = TRUE)
  mse_cols <- grep("_MSE$", names(mse$ind), value = TRUE)
  for (col in mse_cols) {
    expect_true(all(mse$ind[[col]] >= 0), info = paste("MSE:", col))
  }
})
