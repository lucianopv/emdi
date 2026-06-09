#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// Spatial EBLUP core (dense). A = inv((I-rho W')(I-rho W)); G = sigma2_u A;
// V = G + diag(vardir); beta = (X'V^-1 X)^-1 X'V^-1 y; u = G V^-1 (y - X beta).
// Transcription of eblup_SFH (R/eblup.R:87). Returns beta_hat, Q, u_hat, V, Vi.
// [[Rcpp::export]]
Rcpp::List fh_eblup_sfh_cpp(double sigmau2, double rho,
                            const arma::vec& direct, const arma::mat& X,
                            const arma::vec& vardir, const arma::mat& W) {
  const arma::uword m = direct.n_elem;
  arma::mat I  = arma::eye(m, m);
  arma::mat Wt = W.t();
  arma::mat A  = arma::inv((I - rho * Wt) * (I - rho * W));
  arma::mat G  = sigmau2 * A;
  arma::mat V  = G + arma::diagmat(vardir);
  arma::mat Vi = arma::inv(V);
  arma::mat Q  = arma::inv(X.t() * Vi * X);
  arma::vec beta_hat = Q * X.t() * Vi * direct;
  arma::vec res = direct - X * beta_hat;
  arma::vec u_hat = G * Vi * res;
  // Return Vi too so the R wrapper need not re-solve the dense V (the whole
  // point of the cpp path); G is recovered cheaply in R as V - diag(vardir).
  return Rcpp::List::create(Rcpp::Named("beta_hat") = beta_hat,
                            Rcpp::Named("Q") = Q,
                            Rcpp::Named("u_hat") = u_hat,
                            Rcpp::Named("V") = V,
                            Rcpp::Named("Vi") = Vi);
}
