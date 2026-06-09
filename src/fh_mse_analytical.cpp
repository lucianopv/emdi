#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// Prasad-Rao analytical MSE for the standard Fay-Herriot model (method = reml).
// Diagonal V; g1 = psi*(1-B), g2 = B^2 x'Qx.
// VarA = 2/sum(vi^2); g3 = B^2 * VarA / (s2+psi); mse = g1 + g2 + 2 g3.
// Out-of-sample: s2 + x_out' Q x_out.
// Mirrors prasad_rao (R/mse.R). pred_X may have 0 rows.
// [[Rcpp::export]]
Rcpp::List fh_mse_pr_cpp(double sigmau2,
                         const arma::mat& X,
                         const arma::vec& vardir,
                         const arma::mat& pred_X) {
  const arma::uword m = vardir.n_elem;
  const arma::uword p = X.n_cols;
  if (X.n_rows != m) Rcpp::stop("fh_mse_pr_cpp: nrow(X) != length(vardir)");
  if (pred_X.n_cols != p) Rcpp::stop("fh_mse_pr_cpp: ncol(pred_X) != ncol(X)");

  arma::vec vi = 1.0 / (sigmau2 + vardir);
  arma::vec Bd = vardir / (sigmau2 + vardir);
  double SumAD2 = arma::accu(vi % vi);
  double VarA = 2.0 / SumAD2;

  // X' diag(vi) X — build column-by-column for clarity
  arma::mat XtViX(p, p, arma::fill::zeros);
  for (arma::uword i = 0; i < m; ++i) {
    arma::rowvec xi = X.row(i);
    XtViX += vi(i) * (xi.t() * xi);
  }
  arma::mat Q = arma::inv_sympd(XtViX);

  arma::vec mse_in(m);
  for (arma::uword d = 0; d < m; ++d) {
    arma::rowvec xd = X.row(d);
    double g1 = vardir(d) * (1.0 - Bd(d));
    double g2 = Bd(d) * Bd(d) * arma::as_scalar(xd * Q * xd.t());
    double g3 = Bd(d) * Bd(d) * VarA / (sigmau2 + vardir(d));
    mse_in(d) = g1 + g2 + 2.0 * g3;
  }

  arma::vec mse_out(pred_X.n_rows);
  for (arma::uword d = 0; d < pred_X.n_rows; ++d) {
    arma::rowvec xo = pred_X.row(d);
    mse_out(d) = sigmau2 + arma::as_scalar(xo * Q * xo.t());
  }

  return Rcpp::List::create(
    Rcpp::Named("mse_in")  = mse_in,
    Rcpp::Named("mse_out") = mse_out
  );
}
