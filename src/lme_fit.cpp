#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
Rcpp::List data_transform_cpp(const arma::vec& y_raw,
                               const std::string& transformation,
                               double lambda) {
  arma::vec y = y_raw;
  double shift = 0.0;
  bool shift_is_null = false;

  if (transformation == "no") {
    shift_is_null = true;
  } else if (transformation == "log") {
    double mn = y.min();
    if (mn <= 0) { shift = std::abs(mn) + 1.0; y = y + shift; }
    y = arma::log(y);
  } else if (transformation == "box.cox") {
    double mn = y.min();
    if (mn <= 0) { shift = shift + std::abs(mn) + 1.0; }
    if (std::abs(lambda) <= 1e-12) { y = arma::log(y + shift); }
    else { y = (arma::pow(y + shift, lambda) - 1.0) / lambda; }
  } else if (transformation == "dual") {
    double mn = y.min();
    if (mn <= 0) { shift = shift + std::abs(mn) + 1.0; }
    if (std::abs(lambda) <= 1e-12) { y = arma::log(y + shift); }
    else {
      arma::vec yps = y + shift;
      y = (arma::pow(yps, lambda) - arma::pow(yps, -lambda)) / (2.0 * lambda);
    }
  } else if (transformation == "log.shift") {
    double mn = arma::min(y + lambda);
    if (mn <= 0) { lambda = lambda + std::abs(mn) + 1.0; }
    y = arma::log(y + lambda);
    shift_is_null = true;
  } else {
    Rcpp::stop("Unknown transformation: " + transformation);
  }

  if (shift_is_null) {
    return Rcpp::List::create(Rcpp::Named("y") = y, Rcpp::Named("shift") = R_NilValue);
  } else {
    return Rcpp::List::create(Rcpp::Named("y") = y, Rcpp::Named("shift") = shift);
  }
}
