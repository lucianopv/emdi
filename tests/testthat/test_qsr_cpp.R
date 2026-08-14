# Regression tests for the C++ Quintile_Share (QSR) kernel.
#
# The unweighted branch of the QSR kernel in src/indicators.cpp previously used
# type-7 linear interpolation for the 0.2/0.8 quantiles. R's qsr() (see
# R/framework_ebp.R) calls wtd.quantile() (see R/framework_direct.R) regardless
# of whether the weights are all one, and that is the inverse-CDF ("step") rule.
# Interpolating shifts the quintile cut, which changes *which* observations fall
# into the bottom/top quintile -- on eusilcA this moved Quintile_Share by up to
# 8.5% in 52 of 94 domains. The kernel now uses the step rule (located by binary
# search) for the unweighted case.
#
# Note the asymmetry these tests pin down: qsr() uses the step rule always,
# while quants() (Quantile_10..Quantile_90) special-cases unit weights to
# stats::quantile() (type-7). Both conventions must be preserved.

# Reference implementation, mirroring R/framework_ebp.R's qsr() exactly.
# wtd.quantile() is the package's own internal definition.
qsr_reference <- function(y, pop_weights) {
  quant14 <- wtd.quantile(x = y, weights = pop_weights, probs = c(0.2, 0.8))
  iq1 <- y <= quant14[1]
  iq4 <- y > quant14[2]
  as.numeric((sum(pop_weights[iq4] * y[iq4]) / sum(pop_weights[iq4])) /
               (sum(pop_weights[iq1] * y[iq1]) / sum(pop_weights[iq1])))
}

MASK_ALL <- 0x3FFL
MASK_GINI <- 0x008L
MASK_QSR <- 0x010L

test_that("unweighted QSR kernel matches the R step-quantile reference", {
  set.seed(20240101)
  # n = 47 is a case where type-7 and the step rule select different
  # observations; the multiples of 5 exercise the rw[sel] == p tie branch.
  for (n in c(7, 20, 33, 47, 99, 100, 101, 250, 500, 1000, 1237, 5000)) {
    y <- rlnorm(n, meanlog = 10, sdlog = 0.6)
    w <- rep(1, n)
    expect_equal(
      compute_domain_indicators_cpp(y, w, 10859.24)[5],
      qsr_reference(y, w),
      tolerance = 1e-12,
      info = paste("n =", n)
    )
  }
})

test_that("unweighted QSR kernel matches the reference with ties in the data", {
  set.seed(20240102)
  for (n in c(50, 137, 1000)) {
    y <- round(rlnorm(n, meanlog = 6, sdlog = 0.6))  # integer-valued -> many ties
    w <- rep(1, n)
    expect_equal(
      compute_domain_indicators_cpp(y, w, 400)[5],
      qsr_reference(y, w),
      tolerance = 1e-12,
      info = paste("n =", n)
    )
  }
})

test_that("weighted QSR kernel still matches the R step-quantile reference", {
  set.seed(20240103)
  for (n in c(37, 100, 1000)) {
    y <- rlnorm(n, meanlog = 10, sdlog = 0.6)
    w <- runif(n, 0.5, 5)
    expect_equal(
      compute_domain_indicators_cpp(y, w, 10859.24)[5],
      qsr_reference(y, w),
      tolerance = 1e-12,
      info = paste("n =", n)
    )
  }
})

test_that("QSR is invariant to which other indicators are requested", {
  # need_full_sort is TRUE when Gini is requested and FALSE otherwise, which
  # routes the unweighted QSR through two different code paths. They must agree.
  set.seed(20240104)
  for (n in c(47, 100, 1000)) {
    y <- rlnorm(n, meanlog = 10, sdlog = 0.6)
    w <- rep(1, n)
    with_gini <- compute_domain_indicators_selective_cpp(
      y, w, 10859.24, bitwOr(MASK_QSR, MASK_GINI))[5]
    without_gini <- compute_domain_indicators_selective_cpp(
      y, w, 10859.24, MASK_QSR)[5]
    expect_equal(with_gini, without_gini, tolerance = 1e-12,
                 info = paste("n =", n))
    expect_equal(with_gini, qsr_reference(y, w), tolerance = 1e-12,
                 info = paste("n =", n))
  }
})

test_that("Quantile_10..Quantile_90 keep the type-7 convention for unit weights", {
  # Guard against the step rule leaking into quants(), which special-cases
  # unit weights to stats::quantile() (type-7). See R/framework_ebp.R.
  set.seed(20240105)
  for (n in c(47, 100, 1000)) {
    y <- rlnorm(n, meanlog = 10, sdlog = 0.6)
    w <- rep(1, n)
    expect_equal(
      compute_domain_indicators_cpp(y, w, 10859.24)[6:10],
      as.numeric(quantile(y, probs = c(0.10, 0.25, 0.50, 0.75, 0.90),
                          names = FALSE)),
      tolerance = 1e-12,
      info = paste("n =", n)
    )
  }
})

test_that("multi-domain QSR path matches the reference per domain", {
  set.seed(20240106)
  n_dom <- 12
  sizes <- sample(30:200, n_dom, replace = TRUE)
  y <- unlist(lapply(sizes, function(k) rlnorm(k, 10, 0.6)))
  domain_ids <- rep.int(seq_len(n_dom), sizes)
  w <- rep(1, length(y))

  # domain_ids are 1-based and must be contiguous (see compute_all_indicators_cpp)
  got <- compute_all_indicators_cpp(y, w, as.integer(domain_ids),
                                    10859.24, n_dom)[, 5]
  want <- vapply(seq_len(n_dom), function(d) {
    yd <- y[domain_ids == d]
    qsr_reference(yd, rep(1, length(yd)))
  }, numeric(1))
  expect_equal(got, want, tolerance = 1e-12)
})

test_that("ebp() reproduces the released-emdi reference implementation", {
  skip_on_cran()

  # The oracle is a STORED fixture, not a live call into released emdi.
  #
  # It used to be the latter, resolved at run time via
  # getExportedValue("emdi", "ebp"), back when this package was still called
  # emdi2 and "emdi" therefore meant the released CRAN package. It now IS emdi,
  # so that lookup would resolve to the package under test: the oracle becomes
  # the subject and the assertion compares the kernel against itself. A test
  # that cannot fail is worse than no test, so the reference was frozen before
  # the rename rather than deleted after it.
  #
  # EBP/ebp_indicators_emdi223.csv holds ebp()$ind from CRAN emdi 2.2.3 for the
  # argument list below, written at %.17g so it round-trips bit-exactly and
  # contributes no error of its own. ebp() defaults to seed = 123, so no
  # set.seed() is needed for reproducibility. Regenerate only against a genuine
  # released emdi, never against this package.
  want <- read.csv(test_path("EBP", "ebp_indicators_emdi223.csv"),
                   stringsAsFactors = FALSE)

  data("eusilcA_pop", package = "emdi")
  data("eusilcA_smp", package = "emdi")
  fixed <- eqIncome ~ gender + eqsize + cash + self_empl + unempl_ben +
    age_ben + surv_ben + sick_ben + dis_ben + rent + fam_allow +
    house_allow + cap_inv + tax_adj

  got <- ebp(fixed = fixed, pop_data = eusilcA_pop, pop_domains = "district",
             smp_data = eusilcA_smp, smp_domains = "district",
             threshold = 10859.24, transformation = "no", L = 20,
             MSE = FALSE)$ind

  expect_identical(as.character(got$Domain), as.character(want$Domain))

  # Quintile_Share is what this file is about: the type-7-vs-step regression
  # showed up here as 52 of 94 domains differing by up to 2.9%.
  expect_equal(got$Quintile_Share, want$Quintile_Share, tolerance = 1e-8)

  # The fixture carries the other nine indicators too, so assert them as well
  # -- they cost nothing extra and turn a single-indicator check into a full
  # point-estimate parity benchmark against released emdi. All ten agree to
  # ~1e-15 relative, so 1e-8 is a loose bar that only a real defect trips.
  for (nm in setdiff(names(want), c("Domain", "Quintile_Share"))) {
    expect_equal(got[[nm]], want[[nm]], tolerance = 1e-8, info = nm)
  }
})

# --- empty top quintile -------------------------------------------------------
# Quintile_Share is undefined when the 80th percentile equals the domain
# maximum: qsr() uses `iq4 <- y > q[0.8]` (strict) against `iq1 <- y <= q[0.2]`,
# so the TOP quintile is empty and the ratio computes 0/0. Under the step rule
# q[0.8] is an order statistic, so ties at the top -- or a constant domain --
# reach it.
#
# The C++ kernel used to guard the division and return 0, a plausible-looking
# number that averages and plots silently. R returns NaN. These pin the two to
# the same answer. The underlying asymmetry is inherited from upstream emdi
# (byte-identical qsr()) and is deliberately NOT fixed here.

test_that("C++ Quintile_Share is NaN, not 0, when the top quintile is empty", {
  degenerate <- list(c(1, 2, 3, 4, 4), c(1, 1, 1, 1, 1), c(1, 2, 3, 3, 3))
  for (y in degenerate) {
    got <- compute_domain_indicators_cpp(as.numeric(y), rep(1, length(y)), 3)[5]
    expect_true(is.nan(got),
                info = paste("y =", paste(y, collapse = ",")))
  }
})

test_that("C++ and R agree on the empty-top-quintile case", {
  qsr_r <- function(y, w) {
    q <- wtd.quantile(x = y, weights = w, probs = c(0.2, 0.8))
    iq1 <- y <= q[1]; iq4 <- y > q[2]
    as.numeric((sum(w[iq4] * y[iq4]) / sum(w[iq4])) /
               (sum(w[iq1] * y[iq1]) / sum(w[iq1])))
  }
  for (y in list(c(1, 2, 3, 4, 5), c(1, 2, 3, 4, 4), c(1, 1, 1, 1, 1))) {
    w <- rep(1, length(y))
    cpp <- compute_domain_indicators_cpp(as.numeric(y), w, 3)[5]
    r <- qsr_r(as.numeric(y), w)
    expect_identical(is.nan(cpp), is.nan(r),
                     info = paste("y =", paste(y, collapse = ",")))
    if (!is.nan(r)) expect_equal(cpp, r, tolerance = 1e-12)
  }
})
