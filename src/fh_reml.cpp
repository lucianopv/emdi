#include <RcppArmadillo.h>
#include <cmath>
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
