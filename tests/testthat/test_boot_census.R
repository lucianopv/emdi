# The opt-in bootstrap census: options(emdi.boot_census = "model").
#
# emdi's parametric bootstrap census gives the households of out-of-sample
# domains errors N(0, sigma_e^2 + sigma_u^2) on top of the domain effect
# u*_d ~ N(0, sigma_u^2) that every domain already draws, so sigma_u^2 enters
# those households twice. Molina and Rao (2010, section 6, eqs. 18-19) draw
# N(0, sigma_e^2) for every household. The option switches the census to
# theirs; unset, nothing changes. The census is built in two places, and the
# option must reach both: parametric_bootstrap_cpp() on the default engine's
# fast path (standard indicators, numeric threshold) and superpopulation() ->
# gen_superpop_cpp() on the R fallback. A switch in superpopulation() alone
# leaves the default engine's MSE bit-identical, which these tests catch.
#
# The toy puts the sampled domains first in pop_data's (sorted) order. Both
# census builders draw the in-sample households' errors first, then the
# out-of-sample ones, then the domain effects, so the two censuses get the
# same normals household by household and differ only in the out-of-sample
# households' scale. Everything after the census (effects, bootstrap samples,
# refits, bootstrap EBPs) is then shared.

boot_census_toy <- function() {
  set.seed(20261123)
  D <- 30
  D_in <- 20
  N_d <- rep(c(150, 200, 250), length.out = D)
  dom <- factor(rep(sprintf("d%02d", seq_len(D)), N_d))
  u <- rnorm(D, 0, sqrt(0.2))
  x1 <- rnorm(length(dom))
  x2 <- rbinom(length(dom), 1, 0.4)
  y <- exp(9 + 0.4 * x1 + 0.3 * x2 + rep(u, N_d) +
             rnorm(length(dom), 0, sqrt(0.5)))
  full <- data.frame(dom = dom, x1 = x1, x2 = x2, y = y)
  smp <- unlist(lapply(sprintf("d%02d", seq_len(D_in)), function(d) {
    sample(which(full$dom == d), 12)
  }))
  # The population is shuffled; framework_ebp() sorts it by domain, which puts
  # d01-d20 (the sampled ones) first.
  list(pop = full[sample(nrow(full)), c("dom", "x1", "x2")],
       smp = full[smp, ],
       line = unname(stats::median(y)))
}

run_boot_census <- function(census = NULL, engine = NULL) {
  d <- boot_census_toy()
  withr::with_options(list(emdi.boot_census = census, emdi.engine = engine), {
    fit <- NULL
    utils::capture.output(fit <- suppressMessages(ebp(
      fixed = y ~ x1 + x2,
      pop_data = d$pop, pop_domains = "dom",
      smp_data = d$smp, smp_domains = "dom",
      threshold = d$line, transformation = "log",
      L = 5, MSE = TRUE, B = 5, seed = 7, cpus = 1
    )))
    fit
  })
}

boot_census_checksum <- function(df) colSums(as.matrix(df[, -1]))

boot_census_oos <- paste0("d", 21:30)

# Recorded from b5712db (before the option existed) on this toy, as
# sprintf("%.17g") of boot_census_checksum().
boot_census_b5712db <- list(
  auto = list(
    ind = c(
      Mean = 476371.06602197979,
      Head_Count = 14.331733333333334,
      Poverty_Gap = 6.677031108607034,
      Gini = 14.367305953089444,
      Quintile_Share = 401.30962520153463,
      Quantile_10 = 97693.21837317168,
      Quantile_25 = 169938.82709813089,
      Median = 313175.34523070441,
      Quantile_75 = 579608.45728426159,
      Quantile_90 = 1007806.0100877851
    ),
    MSE = c(
      Mean = 1093782217.9371731,
      Head_Count = 0.47227974844444442,
      Poverty_Gap = 0.17957302952001139,
      Gini = 0.045890213690575214,
      Quintile_Share = 354.06991561256541,
      Quantile_10 = 36984706.868422233,
      Quantile_25 = 104788490.26761708,
      Median = 419687540.96869642,
      Quantile_75 = 1656170717.7860169,
      Quantile_90 = 5802384082.3994761
    )
  ),
  r = list(
    ind = c(
      Mean = 476371.06602197979,
      Head_Count = 14.331733333333334,
      Poverty_Gap = 6.677031108607034,
      Gini = 14.367305953089444,
      Quintile_Share = 401.30962520153463,
      Quantile_10 = 97693.21837317168,
      Quantile_25 = 169938.82709813089,
      Median = 313175.34523070441,
      Quantile_75 = 579608.45728426159,
      Quantile_90 = 1007806.0100877851
    ),
    MSE = c(
      Mean = 1159946151.0253484,
      Head_Count = 0.52354984533333326,
      Poverty_Gap = 0.19832946978420363,
      Gini = 0.048157388457297681,
      Quintile_Share = 365.17776831332725,
      Quantile_10 = 40522817.426445633,
      Quantile_25 = 112224763.07452302,
      Median = 448133517.24882495,
      Quantile_75 = 1739265503.5335052,
      Quantile_90 = 6155889123.8680248
    )
  )
)

test_that("the toy reaches each engine's census builder", {
  skip_on_cran()
  # Prove which path each run exercised: the default engine builds its census
  # in parametric_bootstrap_cpp(), the R engine in gen_superpop_cpp().
  count <- function(engine) {
    n <- c(fast = 0L, superpop = 0L)
    orig_fast <- parametric_bootstrap_cpp
    orig_superpop <- gen_superpop_cpp
    testthat::with_mocked_bindings(
      invisible(run_boot_census(engine = engine)),
      parametric_bootstrap_cpp = function(...) {
        n[["fast"]] <<- n[["fast"]] + 1L
        orig_fast(...)
      },
      gen_superpop_cpp = function(...) {
        n[["superpop"]] <<- n[["superpop"]] + 1L
        orig_superpop(...)
      },
      .package = "emdi"
    )
    n
  }
  expect_equal(count(NULL), c(fast = 1L, superpop = 0L))
  expect_equal(count("r"), c(fast = 0L, superpop = 5L))
})

test_that("default engine: unset is b5712db's; 'model' moves only out-of-sample MSE", {
  skip_on_cran()
  emdi_census <- run_boot_census()
  model_census <- run_boot_census(census = "model")

  # The default stays upstream's behaviour.
  expect_equal(boot_census_checksum(emdi_census$ind),
               boot_census_b5712db$auto$ind, tolerance = 1e-10)
  expect_equal(boot_census_checksum(emdi_census$MSE),
               boot_census_b5712db$auto$MSE, tolerance = 1e-10)

  # The census never enters the point estimates.
  expect_identical(model_census$ind, emdi_census$ind)

  # In-sample domains: same census draws, so the effects, bootstrap samples,
  # refits and bootstrap EBPs are shared and their MSE is bit-identical.
  ins <- !emdi_census$MSE$Domain %in% boot_census_oos
  expect_equal(sum(ins), 20L)
  expect_identical(model_census$MSE[ins, ], emdi_census$MSE[ins, ])

  # Out of sample: emdi's census (household variance sigma_e^2 + 2 sigma_u^2)
  # puts the level mean's truth above the EBP in every replicate, so its mean
  # MSE exceeds the model census's; its truths' head counts vary less across
  # replicates, so its head-count MSE falls below.
  oos <- !ins
  expect_gt(sum(emdi_census$MSE$Mean[oos]), sum(model_census$MSE$Mean[oos]))
  expect_lt(sum(emdi_census$MSE$Head_Count[oos]),
            sum(model_census$MSE$Head_Count[oos]))
})

test_that("R engine: unset is b5712db's; 'model' moves out-of-sample MSE", {
  skip_on_cran()
  emdi_census <- run_boot_census(engine = "r")
  model_census <- run_boot_census(census = "model", engine = "r")

  expect_equal(boot_census_checksum(emdi_census$ind),
               boot_census_b5712db$r$ind, tolerance = 1e-10)
  expect_equal(boot_census_checksum(emdi_census$MSE),
               boot_census_b5712db$r$MSE, tolerance = 1e-10)

  expect_identical(model_census$ind, emdi_census$ind)

  oos <- emdi_census$MSE$Domain %in% boot_census_oos
  expect_equal(sum(oos), 10L)
  expect_gt(sum(emdi_census$MSE$Mean[oos]), sum(model_census$MSE$Mean[oos]))
  expect_lt(sum(emdi_census$MSE$Head_Count[oos]),
            sum(model_census$MSE$Head_Count[oos]))
})
