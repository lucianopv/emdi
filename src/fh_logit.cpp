#include <RcppArmadillo.h>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
// [[Rcpp::depends(RcppArmadillo)]]

#ifdef _OPENMP
  #include <omp.h>
#endif

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

// ---------------------------------------------------------------------------
// Full logit parametric-bootstrap MSE loop, mirroring fh_boot_arcsin_cpp.
//
// Per iteration: draw the true value on the transformed scale, form the
// bootstrap sample, re-estimate sigmau2 by REML, refit the EBLUP, back-
// transform (naive or bias-corrected), and accumulate. Returns mse, Li, Ui.
//
// RNG is supplied as pre-generated M x B / m x B matrices rather than drawn
// here, exactly as the arcsin kernel does: R::rnorm is not thread-safe, and
// generating up front is what makes the result independent of thread count.
// The consequence, also shared with arcsin, is that this does NOT reproduce
// boot_logit()'s stream draw-for-draw -- R interleaves its draws with a helper
// rnorm(1) per iteration. Agreement with the R loop is statistical, not exact;
// agreement with the same draws is exact, which is what the tests assert.
//
// One structural difference from arcsin: no truncation. arcsin clamps the
// transformed true value to [0, pi/2] because sin^2 is only invertible there;
// the logit scale is unbounded, so boot_logit does not clamp and neither does
// this.
// ---------------------------------------------------------------------------

// From fh_reml.cpp
double fh_estsigmau2_reml_cpp(const arma::vec& direct, const arma::mat& X,
                              const arma::vec& vardir,
                              double lower, double upper, double tol);

// Type-7 quantile (R's default), matching the arcsin kernel's helper.
static double fh_logit_quantile7(std::vector<double> v, double p) {
  const size_t n = v.size();
  std::sort(v.begin(), v.end());
  double h = (n - 1) * p;
  size_t lo = (size_t) std::floor(h);
  if (lo + 1 >= n) return v[n - 1];
  return v[lo] + (h - lo) * (v[lo + 1] - v[lo]);
}

// [[Rcpp::export]]
Rcpp::List fh_boot_logit_cpp(double sigmau2,
                             const arma::vec& vardir,
                             const arma::vec& beta,
                             const arma::mat& X,
                             const arma::mat& predX,
                             const arma::ivec& is_in,
                             const arma::mat& v_boot,
                             const arma::mat& e_boot,
                             const arma::vec& eblup_corr,
                             bool bc,
                             double lower, double upper,
                             int threads = 1) {
  const arma::uword M = predX.n_rows;
  const arma::uword m = X.n_rows;
  const arma::uword p = X.n_cols;
  const arma::uword B = v_boot.n_cols;

  // All validation before the parallel region: the R API must not be touched
  // from an OpenMP worker.
  if (vardir.n_elem != m) Rcpp::stop("fh_boot_logit_cpp: length(vardir) != nrow(X)");
  if (beta.n_elem != p) Rcpp::stop("fh_boot_logit_cpp: length(beta) != ncol(X)");
  if (predX.n_cols != p) Rcpp::stop("fh_boot_logit_cpp: ncol(predX) != ncol(X)");
  if (v_boot.n_rows != M || e_boot.n_rows != m || e_boot.n_cols != B)
    Rcpp::stop("fh_boot_logit_cpp: RNG matrix dimension mismatch");
  if (is_in.n_elem != M) Rcpp::stop("fh_boot_logit_cpp: length(is_in) != M");
  if (eblup_corr.n_elem != M) Rcpp::stop("fh_boot_logit_cpp: length(eblup_corr) != M");
  if (lower < 0.0) Rcpp::stop("fh_boot_logit_cpp: lower must be >= 0");
  if (threads < 1) threads = 1;

  // in-sample positions, ascending
  arma::uvec in_idx(m);
  { arma::uword k = 0; for (arma::uword d = 0; d < M; ++d) if (is_in(d) == 1) in_idx(k++) = d; }

  const std::vector<double>& gx = fh_gl_x_shared();
  const std::vector<double>& gw = fh_gl_w_shared();
  const double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.25);

  // Loop-invariant: predX %*% beta does not depend on b. boot_logit() rebuilds
  // this (and the whole model matrix, via makeXY) on every iteration.
  const arma::vec Xbeta = predX * beta;

  arma::mat est(M, B), tru(M, B);
  bool boot_failed = false;
  std::string boot_errmsg;

  #ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(threads)
  #endif
  for (long b = 0; b < (long) B; ++b) {
    if (boot_failed) continue;
    try {
      const arma::vec vb = v_boot.col(b);
      const arma::vec eb = e_boot.col(b);

      // True value on the transformed scale, back-transformed naively. No
      // clamping: the logit scale is unbounded (see header note).
      arma::vec tt = Xbeta + vb;
      arma::vec tcol(M);
      for (arma::uword d = 0; d < M; ++d) tcol(d) = logit_inv(tt(d));
      tru.col(b) = tcol;

      arma::vec ystar(m);
      for (arma::uword i = 0; i < m; ++i)
        ystar(i) = Xbeta(in_idx(i)) + vb(in_idx(i)) + eb(i);

      const double s2b = fh_estsigmau2_reml_cpp(ystar, X, vardir, lower, upper, tol);

      const arma::vec vi = 1.0 / (s2b + vardir);
      arma::mat XtViX(p, p, arma::fill::zeros);
      arma::vec XtViy(p, arma::fill::zeros);
      for (arma::uword i = 0; i < m; ++i) {
        const arma::rowvec xi = X.row(i);
        XtViX += vi(i) * (xi.t() * xi);
        XtViy += vi(i) * xi.t() * ystar(i);
      }
      const arma::vec bb = arma::solve(XtViX, XtViy);
      const arma::vec uh = s2b * (vi % (ystar - X * bb));

      arma::vec est_trans = predX * bb;                 // out-of-sample default
      for (arma::uword i = 0; i < m; ++i)
        est_trans(in_idx(i)) = arma::as_scalar(X.row(i) * bb) + uh(i);

      // sd on the transformed scale; zero outside the sample, which makes the
      // bias-corrected branch fall back to the naive value there -- the same
      // thing logit_bc() does for !obs_dom.
      arma::vec sd_all(M, arma::fill::zeros);
      for (arma::uword i = 0; i < m; ++i)
        sd_all(in_idx(i)) = std::sqrt(s2b * (vardir(i) / (s2b + vardir(i))));

      arma::vec ev(M);
      for (arma::uword d = 0; d < M; ++d) {
        if (bc && sd_all(d) > 0.0) {
          ev(d) = fh_logit_one(est_trans(d), sd_all(d), gx, gw);
        } else {
          ev(d) = logit_inv(est_trans(d));
        }
      }
      est.col(b) = ev;
    } catch (std::exception& e) {
      #ifdef _OPENMP
      #pragma omp critical (fh_boot_logit_err)
      #endif
      { boot_failed = true; boot_errmsg = e.what(); }
    } catch (...) {
      #ifdef _OPENMP
      #pragma omp critical (fh_boot_logit_err)
      #endif
      { boot_failed = true; boot_errmsg = "unknown error"; }
    }
  }

  if (boot_failed) Rcpp::stop("fh_boot_logit_cpp: " + boot_errmsg);

  arma::vec mse(M), Li(M), Ui(M);
  std::vector<double> d_row(B);
  for (arma::uword d = 0; d < M; ++d) {
    double acc = 0.0;
    for (arma::uword b = 0; b < B; ++b) {
      const double diff = est(d, b) - tru(d, b);
      d_row[b] = diff;
      acc += diff * diff;
    }
    mse(d) = acc / (double) B;
    Li(d) = eblup_corr(d) + fh_logit_quantile7(d_row, 0.025);
    Ui(d) = eblup_corr(d) + fh_logit_quantile7(d_row, 0.975);
  }

  return Rcpp::List::create(Rcpp::Named("mse") = mse,
                            Rcpp::Named("Li") = Li,
                            Rcpp::Named("Ui") = Ui);
}
