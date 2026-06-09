# Oracle: the exact integrand boot_arcsin_2 / arcsin_bc integrate numerically.
# Use tight integration bounds [max(0, mu-8s), min(pi/2, mu+8s)] so that
# integrate() does not miss the sharp spike when sigma is very small.
bc_integrand <- function(x, mean, sd) sin(x)^2 * stats::dnorm(x, mean = mean, sd = sd)
bc_ref <- function(mu, s) {
  lo <- max(0,    mu - 8 * s)
  hi <- min(pi/2, mu + 8 * s)
  if (hi <= lo) return(0.0)
  stats::integrate(bc_integrand, lo, hi, mu, s, subdivisions = 1000L)$value
}

test_that("fh_bc_integral_cpp matches R integrate() across a mu/sigma grid", {
  mus    <- c(0.01, 0.2, 0.5, 0.785, 1.2, 1.56, -0.1, 1.7)  # incl. outside [0,pi/2]
  sigmas <- c(1e-4, 1e-3, 0.01, 0.05, 0.1, 0.3, 0.7, 1.0)
  grid <- expand.grid(mu = mus, s = sigmas)
  ref  <- mapply(bc_ref, grid$mu, grid$s)
  got  <- fh_bc_integral_cpp(grid$mu, grid$s)
  expect_equal(as.numeric(got), as.numeric(ref), tolerance = 1e-6)
})

test_that("fh_bc_integral_cpp small-sigma guard returns the correct limit", {
  # sigma -> 0: integral -> sin^2(mu) for mu in (0, pi/2); 0 if mu outside [0,pi/2].
  mu  <- c(0.3, -0.2, 1.7)
  exp <- c(sin(0.3)^2, 0, 0)
  got <- fh_bc_integral_cpp(mu, rep(1e-13, 3))
  expect_equal(as.numeric(got), exp, tolerance = 1e-12)
})
