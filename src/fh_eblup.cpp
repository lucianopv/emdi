#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// Diagonal EBLUP core for the standard Fay-Herriot model.
// V^-1 = diag(1/(sigmau2 + vardir)); u_hat = sigmau2 * V^-1 * (y - X beta).
// Replaces the O(m^3) solve(V) block of eblup_FH with O(m p^2) accumulation.
// Returns beta_hat (p x 1), Q = (X'V^-1 X)^-1 (p x p), u_hat (m x 1).
// [[Rcpp::export]]
Rcpp::List fh_eblup_core_cpp(double sigmau2,
                             const arma::vec& direct,
                             const arma::mat& X,
                             const arma::vec& vardir) {
  const arma::uword m = direct.n_elem;
  const arma::uword p = X.n_cols;

  arma::vec vi = 1.0 / (sigmau2 + vardir);

  arma::mat XtViX(p, p, arma::fill::zeros);
  arma::vec XtViy(p, arma::fill::zeros);
  for (arma::uword i = 0; i < m; ++i) {
    arma::rowvec xi = X.row(i);
    XtViX += vi(i) * (xi.t() * xi);
    XtViy += vi(i) * xi.t() * direct(i);
  }
  arma::mat Q = arma::inv_sympd(XtViX);
  arma::vec beta_hat = Q * XtViy;

  arma::vec res   = direct - X * beta_hat;
  arma::vec u_hat = sigmau2 * (vi % res);          // sigmau2 * V^-1 * res

  return Rcpp::List::create(
    Rcpp::Named("beta_hat") = beta_hat,
    Rcpp::Named("Q")        = Q,
    Rcpp::Named("u_hat")    = u_hat
  );
}
