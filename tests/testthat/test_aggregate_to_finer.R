test_that("aggregate_to rejects a column that doesn't nest within pop_domains", {
  data("eusilcA_smp", package = "emdi")
  data("eusilcA_pop", package = "emdi")

  set.seed(1)
  # Deliberately break nesting: assign a random "cell" label independent of district
  eusilcA_pop$cell <- sample(1:5, nrow(eusilcA_pop), replace = TRUE)

  expect_error(
    ebp(
      fixed = eqIncome ~ gender + eqsize,
      pop_data = eusilcA_pop, pop_domains = "district",
      smp_data = eusilcA_smp, smp_domains = "district",
      threshold = 10924.32, na.rm = TRUE, L = 2,
      aggregate_to = "cell"
    ),
    "does not nest within"
  )
})

test_that("aggregate_to does not change fitted model parameters (betas, sigmau2, sigmae2)", {
  data("eusilcA_smp", package = "emdi")
  data("eusilcA_pop", package = "emdi")

  # A genuinely finer, properly-nested grouping: split each district into 2 halves by row order
  eusilcA_pop$subunit <- paste0(eusilcA_pop$district, "_",
                                 rep(1:2, length.out = nrow(eusilcA_pop)))

  set.seed(42)
  plain <- ebp(
    fixed = eqIncome ~ gender + eqsize,
    pop_data = eusilcA_pop, pop_domains = "district",
    smp_data = eusilcA_smp, smp_domains = "district",
    threshold = 10924.32, na.rm = TRUE, L = 2, MSE = FALSE
  )

  set.seed(42)
  finer <- ebp(
    fixed = eqIncome ~ gender + eqsize,
    pop_data = eusilcA_pop, pop_domains = "district",
    smp_data = eusilcA_smp, smp_domains = "district",
    threshold = 10924.32, na.rm = TRUE, L = 2, MSE = FALSE,
    aggregate_to = "subunit"
  )

  expect_equal(as.numeric(nlme::fixed.effects(plain$model)),
               as.numeric(nlme::fixed.effects(finer$model)),
               tolerance = 1e-8)
  expect_equal(plain$model$sigma^2, finer$model$sigma^2, tolerance = 1e-8)
  expect_equal(as.numeric(nlme::VarCorr(plain$model)[1, 1]),
               as.numeric(nlme::VarCorr(finer$model)[1, 1]),
               tolerance = 1e-8)
})

test_that("aggregate_to produces one indicator row per output group, coherent with the plain fit", {
  data("eusilcA_smp", package = "emdi")
  data("eusilcA_pop", package = "emdi")

  eusilcA_pop$subunit <- paste0(eusilcA_pop$district, "_",
                                 rep(1:2, length.out = nrow(eusilcA_pop)))

  set.seed(7)
  plain <- ebp(
    fixed = eqIncome ~ gender + eqsize,
    pop_data = eusilcA_pop, pop_domains = "district",
    smp_data = eusilcA_smp, smp_domains = "district",
    threshold = 10924.32, na.rm = TRUE, L = 50, MSE = FALSE
  )

  set.seed(7)
  finer <- ebp(
    fixed = eqIncome ~ gender + eqsize,
    pop_data = eusilcA_pop, pop_domains = "district",
    smp_data = eusilcA_smp, smp_domains = "district",
    threshold = 10924.32, na.rm = TRUE, L = 50, MSE = FALSE,
    aggregate_to = "subunit"
  )

  expect_equal(nrow(finer$ind), length(unique(eusilcA_pop$subunit)))

  # Population-weight the subunit means back up to district, compare to the plain fit
  finer$ind$district <- sub("_[12]$", "", as.character(finer$ind$Domain))
  wts <- table(eusilcA_pop$subunit)
  finer$ind$w <- as.numeric(wts[as.character(finer$ind$Domain)])
  agg <- stats::aggregate(Mean * w ~ district, data = finer$ind, FUN = sum)
  wsum <- stats::aggregate(w ~ district, data = finer$ind, FUN = sum)
  agg_mean <- agg$`Mean * w` / wsum$w
  names(agg_mean) <- wsum$district

  plain_mean <- plain$ind$Mean
  names(plain_mean) <- as.character(plain$ind$Domain)

  common <- intersect(names(agg_mean), names(plain_mean))
  expect_gt(length(common), 5)
  expect_equal(unname(agg_mean[common]), unname(plain_mean[common]), tolerance = 0.05)
})
