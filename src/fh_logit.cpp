#include <RcppArmadillo.h>
#include <cmath>
#include <vector>
// [[Rcpp::depends(RcppArmadillo)]]

// Gauss-Legendre nodes/weights, built once in fh_arcsin.cpp. Reused here
// rather than duplicated: the quadrature rule is the same, only the integrand
// differs.
const std::vector<double>& fh_gl_x_shared();
const std::vector<double>& fh_gl_w_shared();

// ---------------------------------------------------------------------------
// C++ kernel for the logit back-transformation integral.
//
// Mirrors fh_bc_integral_cpp (arcsin). Upstream's logit_bc()
// (R/back_transformation.R) computes, per in-sample domain,
//
//   E[logit^-1(theta)],  theta ~ N(mu, var)
//
// by calling integrate() over [mu - 50*sd, mu + 50*sd] -- once per domain per
// bootstrap iteration, with the author's own "# Can this be vectorized?" on
// the loop.
//
// Substituting t = (x - mu)/sigma turns that into a fixed integral in
// standardised space,
//
//   E = int_{-50}^{50} logit^-1(mu + sigma*t) * phi(t) dt
//
// which GL-64 evaluates without any adaptive subdivision. The bounds capture
// essentially all the mass (Phi(-50) underflows to 0), so this is the full
// real line for practical purposes -- the same bounds R uses.
//
// NOTE ON SCOPE: unlike the arcsin port, this is a SPEED change only. The
// arcsin version integrated over a FIXED [0, pi/2], which is what let R's
// adaptive integrate() miss a narrow spike and silently underflow (see NEWS).
// logit_bc's bounds already follow the spike, so upstream avoided that trap
// independently. The kernel must therefore AGREE with integrate(), not
// improve on it.
// ---------------------------------------------------------------------------

// Numerically stable logistic. R's logit_inverse() is exp(l)/(1 + exp(l)),
// which is Inf/Inf = NaN once exp(l) overflows (l beyond ~710). This form
// agrees with it wherever R is finite and stays defined elsewhere, which
// matters because the quadrature evaluates the integrand at mu +/- 50*sigma.
static inline double logit_inv(double l) {
  if (l >= 0.0) {
    return 1.0 / (1.0 + std::exp(-l));
  }
  const double e = std::exp(l);
  return e / (1.0 + e);
}

// Standard normal density.
static inline double std_norm_pdf(double t) {
  static const double inv_sqrt_2pi = 0.398942280401432677939946059934;
  return inv_sqrt_2pi * std::exp(-0.5 * t * t);
}

static double fh_logit_one(double mu, double sigma,
                           const std::vector<double>& gx,
                           const std::vector<double>& gw) {
  // sigma -> 0: the normal collapses to a point mass at mu.
  if (!(sigma > 0.0)) return logit_inv(mu);

  // GL-64 on t in [-8, 8], NOT R's [-50, 50].
  //
  // R integrates over mu +/- 50*sd, but its adaptive rule places nodes where
  // the integrand actually lives. A fixed 64-node rule cannot: spreading 64
  // nodes across a width-100 interval leaves ~1.5 between them, while phi(t)
  // has all of its structure inside |t| < 5. The first attempt did exactly
  // that and was wrong by ~0.07 in absolute terms -- most nodes sat where the
  // integrand is zero and the core was under-resolved.
  //
  // Clipping to +/- 8 matches fh_bc_one's convention for arcsin. The mass
  // discarded is Phi(-8) ~ 6e-16, far below the 1e-8 parity target, and the
  // node spacing drops to ~0.25 where it matters.
  const double half = 8.0;
  double acc = 0.0;
  for (size_t k = 0; k < gx.size(); ++k) {
    const double t = half * gx[k];
    acc += gw[k] * logit_inv(mu + sigma * t) * std_norm_pdf(t);
  }
  return acc * half;
}

// Exported vectorised interface.
// [[Rcpp::export]]
arma::vec fh_logit_integral_cpp(const arma::vec& mu, const arma::vec& sigma) {
  if (mu.n_elem != sigma.n_elem)
    Rcpp::stop("fh_logit_integral_cpp: mu and sigma length mismatch");
  const std::vector<double>& gx = fh_gl_x_shared();
  const std::vector<double>& gw = fh_gl_w_shared();
  arma::vec out(mu.n_elem);
  for (arma::uword i = 0; i < mu.n_elem; ++i)
    out(i) = fh_logit_one(mu(i), sigma(i), gx, gw);
  return out;
}
