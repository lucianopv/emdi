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

// fh_arcsin.cpp — back-transform integral for the arcsin EBP/bootstrap.
//
// Provides fh_bc_integral_cpp(mu, sigma): vectorised evaluation of
//   I(mu, sigma) = int_0^{pi/2} sin^2(x) * N(x; mu, sigma^2) dx
// used in arcsin_bc back-transformation and boot_arcsin_2 bootstrap.
//
// Method: Gauss–Legendre quadrature in standardised t-space.
//   Substitute x = mu + sigma*t so the integrand becomes
//     sin^2(mu + sigma*t) * phi(t)   (standard normal density)
//   The [0, pi/2] limits in x map to [A, B] in t where
//     A = max((0 - mu)/sigma, -8),  B = min((pi/2 - mu)/sigma, 8)
//   The [-8, 8] clamps capture >1-1e-15 of the normal mass.
//   GL-64 nodes on [-1,1] are mapped to [A,B] via the usual affine transform.
//
// Internal helpers fh_bc_one, fh_gl_x, fh_gl_w, fh_gauss_legendre are used
// by later tasks (bootstrap loop); keep their names exactly.

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
// Scalar worker: I(mu, sigma) for a single (mu, sigma) pair.
// ---------------------------------------------------------------------------
static double fh_bc_one(double mu, double sigma,
                        const std::vector<double>& gx,
                        const std::vector<double>& gw) {
  const double HALF_PI = M_PI / 2.0;
  if (sigma <= 1e-12) {
    if (mu <= 0.0 || mu >= HALF_PI) return 0.0;   // spike outside [0,pi/2] -> 0
    double s = std::sin(mu);
    return s * s;                                  // spike inside -> sin^2(mu)
  }
  double A = std::max((0.0 - mu) / sigma, -8.0);
  double B = std::min((HALF_PI - mu) / sigma, 8.0);
  if (B <= A) return 0.0;
  double c1 = 0.5 * (B - A), c2 = 0.5 * (B + A);
  double acc = 0.0;
  for (size_t k = 0; k < gx.size(); ++k) {
    double t = c1 * gx[k] + c2;
    double x = mu + sigma * t;
    double s = std::sin(x);
    double phi = std::exp(-0.5 * t * t) / FH_SQRT_2PI;
    acc += gw[k] * s * s * phi;
  }
  return c1 * acc;
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
static const std::vector<double>& fh_gl_x() { return fh_gl().x; }
static const std::vector<double>& fh_gl_w() { return fh_gl().w; }

// ---------------------------------------------------------------------------
// Exported vectorised interface.
// I = int_0^{pi/2} sin^2(x) * N(x; mu, sigma^2) dx, length(mu) == length(sigma).
// [[Rcpp::export]]
arma::vec fh_bc_integral_cpp(const arma::vec& mu, const arma::vec& sigma) {
  if (mu.n_elem != sigma.n_elem)
    Rcpp::stop("fh_bc_integral_cpp: mu and sigma length mismatch");
  const std::vector<double>& gx = fh_gl_x();
  const std::vector<double>& gw = fh_gl_w();
  arma::vec out(mu.n_elem);
  for (arma::uword i = 0; i < mu.n_elem; ++i)
    out(i) = fh_bc_one(mu(i), sigma(i), gx, gw);
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
                              double lower, double upper) {
  const arma::uword M = predX.n_rows;
  const arma::uword m = X.n_rows;
  const arma::uword B = v_boot.n_cols;
  const double HALF_PI = M_PI / 2.0;
  const double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.25);

  if (vardir.n_elem != m) Rcpp::stop("fh_boot_arcsin_cpp: length(vardir) != nrow(X)");
  if (v_boot.n_rows != M || e_boot.n_rows != m || e_boot.n_cols != B)
    Rcpp::stop("fh_boot_arcsin_cpp: RNG matrix dimension mismatch");
  if (is_in.n_elem != M) Rcpp::stop("fh_boot_arcsin_cpp: length(is_in) != M");
  if (lower < 0.0) Rcpp::stop("fh_boot_arcsin_cpp: lower must be >= 0");

  // in-sample index map (positions in 0..M-1), ascending
  arma::uvec in_idx(m);
  { arma::uword k = 0; for (arma::uword d = 0; d < M; ++d) if (is_in(d) == 1) in_idx(k++) = d; }

  // domain position -> in-sample rank (0..m-1), or -1 if out-of-sample
  arma::ivec rank_of(M); rank_of.fill(-1);
  for (arma::uword i = 0; i < m; ++i) rank_of(in_idx(i)) = (arma::sword) i;

  const std::vector<double>& gx = fh_gl_x();
  const std::vector<double>& gw = fh_gl_w();

  arma::vec Xbeta = predX * beta;                 // M, same every iteration
  arma::mat est(M, B), tru(M, B);

  bool boot_failed = false;
  std::string boot_errmsg;

  #ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  for (long b = 0; b < (long) B; ++b) {
    if (boot_failed) continue;                 // skip remaining work after a failure
    try {
      arma::vec vb = v_boot.col(b);
      arma::vec eb = e_boot.col(b);

      arma::vec tt = arma::clamp(Xbeta + vb, 0.0, HALF_PI);
      tru.col(b) = arma::square(arma::sin(tt));

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
          ev(d) = fh_bc_one(mud, std::sqrt(var_in(rank_of(d))), gx, gw);
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
