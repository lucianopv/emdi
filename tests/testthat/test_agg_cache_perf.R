# Regression test for the aggregate-index cache construction in
# src/parametric_bootstrap.cpp.
#
# The cache maps each aggregate_to cell to the population row indices it
# contains. Building it must be O(N_pop) -- a two-pass counting sort -- and in
# particular must NOT depend on the number of aggregate cells. The naive
# construction (one arma::find scan per cell) is O(N_dom_agg * N_pop), which is
# a hard cliff for finer-than-model-domain output: at 3.7M population rows and
# ~140k grid cells it costs minutes before a single bootstrap iteration starts.
#
# See src/monte_carlo.cpp, which already builds the same cache in O(N_pop).

test_that("agg_idx_cache construction does not scale with aggregate cardinality", {
  skip_on_cran()

  data("eusilcA_smp", package = "emdi2")
  data("eusilcA_pop", package = "emdi2")

  # Enlarge the population so the cache cost is clearly separable from the
  # per-iteration bootstrap cost (which is O(B * L * N_pop) and stays small
  # at B = 1, L = 2).
  reps <- 8
  pop <- eusilcA_pop[rep(seq_len(nrow(eusilcA_pop)), reps), ]
  rownames(pop) <- NULL
  # Finest possible aggregation: one output cell per population row, nested
  # inside district. This is the GridSAE-shaped case.
  pop$cell <- seq_len(nrow(pop))

  fixed <- eqIncome ~ gender + eqsize

  framework <- framework_ebp(
    fixed = fixed,
    pop_data = pop, pop_domains = "district",
    smp_data = eusilcA_smp, smp_domains = "district",
    threshold = 10924.32, custom_indicator = NULL,
    na.rm = TRUE, pop_weights = NULL, weights = NULL,
    aggregate_to = "cell"
  )

  set.seed(42)
  pe <- point_estim(
    framework = framework, fixed = fixed,
    transformation = "log", interval = "default", L = 2
  )

  framework$pop_data$eqIncome <- seq_len(nrow(framework$pop_data))
  X_pop <- model.matrix(fixed, framework$pop_data)
  pop_domain_names <- as.character(unique(framework$pop_domains_vec))
  smp_domain_names <- names(table(framework$smp_domains_vec))
  smp_to_pop_map <- match(smp_domain_names, pop_domain_names)
  X_smp <- model.matrix(fixed, framework$smp_data)
  smp_domain_ids <- as.integer(as.factor(framework$smp_data$district))

  agg_ids <- as.integer(framework$aggregate_to_vec)
  N_dom_agg <- framework$N_dom_pop_agg

  expect_gte(N_dom_agg, 2e5)   # guard: the test is meaningless if this shrinks

  set.seed(42)
  elapsed <- system.time(
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
      B = 1L, L = 2L,
      threshold = 10924.32,
      transformation = "log",
      lambda_orig = 0, shift_orig = 0,
      interval_lower = -1, interval_upper = 2,
      agg_domain_ids_pop = agg_ids,
      N_dom_agg = N_dom_agg
    )
  )[["elapsed"]]

  expect_equal(nrow(mse), N_dom_agg)

  # O(N_pop) construction finishes this in ~1s; the O(N_dom_agg * N_pop)
  # construction needs ~25s. 8s separates them with a wide margin either way.
  expect_lt(elapsed, 8)
})
