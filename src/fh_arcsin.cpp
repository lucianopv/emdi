#include <RcppArmadillo.h>
#include <cmath>
#include <vector>
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
