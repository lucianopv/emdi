#include <RcppArmadillo.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif
// [[Rcpp::depends(RcppArmadillo)]]

// fh_arcsin.cpp — bias-corrected back-transformation for the arcsin Fay-Herriot model.
//
// fh_bc_integral_cpp(mu, sigma) evaluates, element by element,
//   I(mu, sigma) = E[sin^2(t)],  t ~ N(mu, sigma^2),  over the WHOLE real line,
// the back-transformation of Hadam, Wuerz, Kreutzmann and Schmid (2023, eq. 5). It has the closed form
//   I = (1 - exp(-2 sigma^2) cos(2 mu)) / 2,
// exact for any mu and sigma (sigma -> 0 gives sin^2(mu)). It is used by arcsin_bc and by the bootstrap
// below. (Before 2026-10-17 this integrated over [0, pi/2] only, losing the mass above pi/2 and pulling
// estimates near 1 down: 0.50 instead of 0.999 at mu = pi/2, sigma = 0.03.)
//
// The Gauss-Legendre table below is no longer used here; src/fh_logit.cpp reuses it through the shared
// accessors.

// ---------------------------------------------------------------------------
// Gauss-Legendre nodes/weights on [-1,1] via Newton iteration on Legendre P_n.
// ---------------------------------------------------------------------------
static void fh_gauss_legendre(int n, std::vector<double>& x, std::vector<double>& w) {
  x.assign(n, 0.0); w.assign(n, 0.0);
  const double eps = 1e-15;
  int m = (n + 1) / 2;
  for (int i = 0; i < m; ++i) {
    double z = std::cos(M_PI * (i + 0.75) / (n + 0.5));
    double z1, pp;
    do {
      double p1 = 1.0, p2 = 0.0;
      for (int j = 0; j < n; ++j) {
        double p3 = p2; p2 = p1;
        p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1.0);
      }
      pp = n * (z * p1 - p2) / (z * z - 1.0);
      z1 = z; z = z1 - p1 / pp;
    } while (std::fabs(z - z1) > eps);
    x[i] = -z;          x[n - 1 - i] = z;
    w[i] = 2.0 / ((1.0 - z * z) * pp * pp);
    w[n - 1 - i] = w[i];
  }
}

static const double FH_SQRT_2PI = 2.506628274631000502415765284811;

// ---------------------------------------------------------------------------
// Scalar worker: I(mu, sigma) = E[sin^2(t)], t ~ N(mu, sigma^2), closed form (Hadam et al. 2023, eq. 5).
// ---------------------------------------------------------------------------
static double fh_bc_one(double mu, double sigma) {
  return 0.5 * (1.0 - std::exp(-2.0 * sigma * sigma) * std::cos(2.0 * mu));
}

// ---------------------------------------------------------------------------
// GL-64 table, initialised exactly once (C++11 thread-safe magic static).
// Safe to call from OpenMP parallel regions (later tasks do).
// ---------------------------------------------------------------------------
struct FhGL {
  std::vector<double> x, w;
  FhGL() { fh_gauss_legendre(64, x, w); }
};
static const FhGL& fh_gl() { static const FhGL g; return g; }

// Non-static accessors so src/fh_logit.cpp can reuse the same quadrature rule
// instead of building a second copy: the nodes are identical, only the
// integrand differs.
const std::vector<double>& fh_gl_x_shared() { return fh_gl().x; }
const std::vector<double>& fh_gl_w_shared() { return fh_gl().w; }

// ---------------------------------------------------------------------------
// Exported vectorised interface.
// I = E[sin^2(t)], t ~ N(mu, sigma^2) over the real line, length(mu) == length(sigma).
// [[Rcpp::export]]
arma::vec fh_bc_integral_cpp(const arma::vec& mu, const arma::vec& sigma) {
  if (mu.n_elem != sigma.n_elem)
    Rcpp::stop("fh_bc_integral_cpp: mu and sigma length mismatch");
  arma::vec out(mu.n_elem);
  for (arma::uword i = 0; i < mu.n_elem; ++i)
    out(i) = fh_bc_one(mu(i), sigma(i));
  return out;
}

// ---------------------------------------------------------------------------
// Defined in src/fh_reml.cpp (Plan 1).
// ---------------------------------------------------------------------------
double fh_estsigmau2_reml_cpp(const arma::vec& direct, const arma::mat& X,
                              const arma::vec& vardir,
                              double lower, double upper, double tol);

// ---------------------------------------------------------------------------
// Type-7 (R default) quantile of a length-n sample (sorts a copy).
// ---------------------------------------------------------------------------
static double fh_quantile7(std::vector<double> v, double p) {
  const size_t n = v.size();
  std::sort(v.begin(), v.end());
  double h = (n - 1) * p;
  size_t lo = (size_t) std::floor(h);
  if (lo + 1 >= n) return v[n - 1];
  return v[lo] + (h - lo) * (v[lo + 1] - v[lo]);
}

// ---------------------------------------------------------------------------
// Full arcsin parametric-bootstrap MSE loop (sequential). is_in: length-M
// 0/1 (1 = in-sample). v_boot: M x B ~ N(0, sigmau2). e_boot: m x B,
// column scaled by sqrt(vardir). Returns mse, Li, Ui (each length M).
// [[Rcpp::export]]
Rcpp::List fh_boot_arcsin_cpp(double sigmau2,                  // reserved: v_boot is pre-scaled ~N(0,sigmau2)
                              const arma::vec& vardir,        // length m
                              const arma::vec& beta,          // length p
                              const arma::mat& X,             // m x p
                              const arma::mat& predX,         // M x p
                              const arma::ivec& is_in,        // length M
                              const arma::mat& v_boot,        // M x B
                              const arma::mat& e_boot,        // m x B
                              const arma::vec& eblup_corr,    // length M
                              bool bc,
                              double lower, double upper,
                              int threads = 1) {
  const arma::uword M = predX.n_rows;
  const arma::uword m = X.n_rows;
  const arma::uword B = v_boot.n_cols;
  const double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.25);

  if (vardir.n_elem != m) Rcpp::stop("fh_boot_arcsin_cpp: length(vardir) != nrow(X)");
  if (v_boot.n_rows != M || e_boot.n_rows != m || e_boot.n_cols != B)
    Rcpp::stop("fh_boot_arcsin_cpp: RNG matrix dimension mismatch");
  if (is_in.n_elem != M) Rcpp::stop("fh_boot_arcsin_cpp: length(is_in) != M");
  if (lower < 0.0) Rcpp::stop("fh_boot_arcsin_cpp: lower must be >= 0");

  if (threads < 1) threads = 1;

  // in-sample index map (positions in 0..M-1), ascending
  arma::uvec in_idx(m);
  { arma::uword k = 0; for (arma::uword d = 0; d < M; ++d) if (is_in(d) == 1) in_idx(k++) = d; }

  // domain position -> in-sample rank (0..m-1), or -1 if out-of-sample
  arma::ivec rank_of(M); rank_of.fill(-1);
  for (arma::uword i = 0; i < m; ++i) rank_of(in_idx(i)) = (arma::sword) i;


  arma::vec Xbeta = predX * beta;                 // M, same every iteration
  arma::mat est(M, B), tru(M, B);

  bool boot_failed = false;
  std::string boot_errmsg;

  #ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(threads)
  #endif
  for (long b = 0; b < (long) B; ++b) {
    if (boot_failed) continue;                 // skip remaining work after a failure
    try {
      arma::vec vb = v_boot.col(b);
      arma::vec eb = e_boot.col(b);

      // bootstrap truth of Hadam et al. (2023, p. 14): sin^2(x'beta + u*), unclipped
      tru.col(b) = arma::square(arma::sin(Xbeta + vb));

      arma::vec ystar(m);
      for (arma::uword i = 0; i < m; ++i) ystar(i) = Xbeta(in_idx(i)) + vb(in_idx(i)) + eb(i);

      double s2b = fh_estsigmau2_reml_cpp(ystar, X, vardir, lower, upper, tol);

      arma::vec vi = 1.0 / (s2b + vardir);
      arma::mat XtViX(X.n_cols, X.n_cols, arma::fill::zeros);
      arma::vec XtViy(X.n_cols, arma::fill::zeros);
      for (arma::uword i = 0; i < m; ++i) {
        arma::rowvec xi = X.row(i);
        XtViX += vi(i) * (xi.t() * xi);
        XtViy += vi(i) * xi.t() * ystar(i);
      }
      arma::mat Q = arma::inv_sympd(XtViX);
      arma::vec bb = Q * XtViy;
      arma::vec uh = s2b * (vi % (ystar - X * bb));

      arma::vec predbeta = predX * bb;
      arma::vec est_trans = predbeta;               // oos default
      for (arma::uword i = 0; i < m; ++i)
        est_trans(in_idx(i)) = arma::as_scalar(X.row(i) * bb) + uh(i);

      arma::vec var_in = s2b * (vardir / (s2b + vardir));  // length m

      arma::vec ev(M);
      for (arma::uword d = 0; d < M; ++d) {
        double mud = est_trans(d);
        if (is_in(d) == 1 && bc) {
          ev(d) = fh_bc_one(mud, std::sqrt(var_in(rank_of(d))));
        } else {
          double s = std::sin(mud); ev(d) = s * s;
        }
      }
      est.col(b) = ev;
    } catch (std::exception& e) {
      #pragma omp critical
      { boot_failed = true; boot_errmsg = e.what(); }
    }
  }

  if (boot_failed)
    Rcpp::stop("fh_boot_arcsin_cpp: bootstrap iteration failed: " + boot_errmsg);

  arma::vec mse(M), Li(M), Ui(M);
  for (arma::uword d = 0; d < M; ++d) {
    arma::rowvec diff = est.row(d) - tru.row(d);
    mse(d) = arma::mean(arma::square(diff));
    std::vector<double> dv(diff.begin(), diff.end());
    Li(d) = eblup_corr(d) + fh_quantile7(dv, 0.025);
    Ui(d) = eblup_corr(d) + fh_quantile7(dv, 0.975);
  }

  return Rcpp::List::create(Rcpp::Named("mse") = mse,
                            Rcpp::Named("Li") = Li,
                            Rcpp::Named("Ui") = Ui);
}
