#include <RcppArmadillo.h>
#include <cmath>
#include <limits>
// [[Rcpp::depends(RcppArmadillo)]]

// Diagonal REML log-likelihood for the standard Fay-Herriot model.
// V = sigmau2 * I + diag(vardir) is diagonal, so V^-1 = diag(1/(sigmau2+vardir))
// and all quantities reduce to O(m * p^2) scalar accumulation. This is the
// closed form of A.reml (R/estim_sigmau2.R) with no M x M solve/eigen/det.
// [[Rcpp::export]]
double fh_reml_loglik_cpp(double sigmau2,
                          const arma::vec& direct,
                          const arma::mat& X,
                          const arma::vec& vardir) {
  const arma::uword m = direct.n_elem;
  const arma::uword p = X.n_cols;

  arma::vec dv = sigmau2 + vardir;          // diagonal of V
  arma::vec vi = 1.0 / dv;                   // diagonal of V^-1

  arma::mat XtViX(p, p, arma::fill::zeros);
  arma::vec XtViy(p, arma::fill::zeros);
  double yViy = 0.0;
  for (arma::uword i = 0; i < m; ++i) {
    arma::rowvec xi = X.row(i);
    XtViX += vi(i) * (xi.t() * xi);
    XtViy += vi(i) * xi.t() * direct(i);
    yViy  += vi(i) * direct(i) * direct(i);
  }
  arma::mat Q = arma::inv_sympd(XtViX);      // (X'V^-1 X)^-1

  // y' P y = y'V^-1 y - (X'V^-1 y)' Q (X'V^-1 y)
  double yPy = yViy - arma::as_scalar(XtViy.t() * Q * XtViy);

  double logdetV = arma::accu(arma::log(dv));            // sum log eigenvalues
  double logdetXtViX, sgn;
  arma::log_det(logdetXtViX, sgn, XtViX);                // log|X'V^-1 X|
  if (sgn <= 0.0) return -1e30;   // singular/indefinite X'V^-1X: sentinel (loglik is maximised)

  return -0.5 * (double)m * std::log(2.0 * M_PI)
         - 0.5 * logdetV
         - 0.5 * logdetXtViX
         - 0.5 * yPy;
}

// Faithful transcription of R's Brent_fmin (src/library/stats/src/optimize.c),
// minimising f on [ax, bx] with the same default tolerance optimize() uses
// (.Machine$double.eps^0.25). Reproduces optimize()'s iteration path so the
// returned argmin matches R to optimizer tolerance.
typedef double (*fh_fn1)(double, void*);

static double fh_brent_fmin(double ax, double bx, fh_fn1 f, void* info, double tol) {
  const double c = (3.0 - std::sqrt(5.0)) * 0.5;
  const double eps = std::sqrt(std::numeric_limits<double>::epsilon());

  double a = ax, b = bx;
  double v = a + c * (b - a), w = v, x = v;
  double d = 0.0, e = 0.0;
  double fx = (*f)(x, info), fv = fx, fw = fx;
  const double tol3 = tol / 3.0;

  for (;;) {
    double xm = (a + b) * 0.5;
    double tol1 = eps * std::fabs(x) + tol3;
    double t2 = 2.0 * tol1;
    if (std::fabs(x - xm) <= t2 - (b - a) * 0.5) break;

    double p = 0.0, q = 0.0, r = 0.0;
    if (std::fabs(e) > tol1) {
      r = (x - w) * (fx - fv);
      q = (x - v) * (fx - fw);
      p = (x - v) * q - (x - w) * r;
      q = (q - r) * 2.0;
      if (q > 0.0) p = -p; else q = -q;
      r = e; e = d;
    }
    if (std::fabs(p) >= std::fabs(0.5 * q * r) ||
        p <= q * (a - x) || p >= q * (b - x)) {
      e = (x < xm) ? (b - x) : (a - x);
      d = c * e;
    } else {
      d = p / q;
      double u_tmp = x + d;
      if (u_tmp - a < t2 || b - u_tmp < t2) {
        d = tol1;
        if (x >= xm) d = -d;
      }
    }
    double u;
    if (std::fabs(d) >= tol1) u = x + d;
    else if (d > 0.0)        u = x + tol1;
    else                     u = x - tol1;

    double fu = (*f)(u, info);
    if (fu <= fx) {
      if (u < x) b = x; else a = x;
      v = w; fv = fw; w = x; fw = fx; x = u; fx = fu;
    } else {
      if (u < x) a = u; else b = u;
      if (fu <= fw || w == x) { v = w; fv = fw; w = u; fw = fu; }
      else if (fu <= fv || v == x || v == w) { v = u; fv = fu; }
    }
  }
  return x;
}

struct fh_reml_data { const arma::vec* direct; const arma::mat* X; const arma::vec* vardir; };

static double fh_neg_reml_loglik(double s2, void* info) {
  fh_reml_data* d = static_cast<fh_reml_data*>(info);
  return -fh_reml_loglik_cpp(s2, *(d->direct), *(d->X), *(d->vardir));
}

// REML estimate of sigmau2 by Brent profiling over [lower, upper], mirroring
// optimize(A.reml, interval, maximum = TRUE).
// [[Rcpp::export]]
double fh_estsigmau2_reml_cpp(const arma::vec& direct,
                              const arma::mat& X,
                              const arma::vec& vardir,
                              double lower, double upper, double tol) {
  if (lower < 0.0) Rcpp::stop("fh_estsigmau2_reml_cpp: lower must be >= 0");
  fh_reml_data info{ &direct, &X, &vardir };
  return fh_brent_fmin(lower, upper, &fh_neg_reml_loglik, &info, tol);
}
