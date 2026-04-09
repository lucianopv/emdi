#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

arma::mat mc_census(arma::mat X, arma::vec betas, arma::vec gamma, arma::vec u_hat, arma::uvec area_id, int L, double sigma_e, double sigma_u, std::string transformation) {
  // Define the dimensions
  int n = X.n_rows;
  int p = X.n_cols;

  arma::uvec levels = arma::unique(area_id);
  int q = levels.n_rows;

  arma::vec u_i = arma::zeros(q);

  // Create u_star depending on the area
  for (int i = 0; i < q; i++) {
      u_i(i) = rnorm(1, 0, sigma_u * std::sqrt((1 - gamma(i))))[0];
  }

  // Create the vector of random effects for each observation
  arma::vec u_star = arma::zeros(n);
  for (int i = 0; i < q; i++) {
    arma::uvec idx = find(area_id == levels(i));
    u_star.elem(idx) += u_i(i);
  }

  arma::vec e_star = rnorm(n, 0, sigma_e);
  arma::mat y_star = X * betas + u_hat + e_star + u_star;
  if (transformation == "log") {
    y_star = arma::exp(y_star);
  }
  y_star = arma::clamp(y_star, 0, arma::datum::inf);
  return y_star;
}

// [[Rcpp::export]]
List mc_census_check(arma::mat X, arma::vec betas, arma::vec gamma, arma::vec u_hat, arma::vec area_id, int L, double sigma_e, double sigma_u, std::string transformation) {
  // Define the dimensions
  int n = X.n_rows;
  int p = X.n_cols;

   arma::vec levels = arma::unique(area_id);
  int q = levels.n_rows;

  arma::vec u_i = arma::zeros(q);

  std::cout << "zero u_i ready and size " << q << std::endl;

  // Create u_star depending on the area
  for (int i = 0; i < q; i++) {
      u_i(i) = rnorm(1, 0, sigma_u * std::sqrt((1 - gamma(i))))[0];
  }

  std::cout << "u_i ready" << std::endl;

  // Create the vector of random effects for each observation
  arma::vec u_star = arma::zeros(n);
  for (int i = 0; i < q; i++) {
    arma::uvec idx = find(area_id == levels(i));
    u_star.elem(idx) += u_i(i);
  }

  arma::vec e_star = rnorm(n, 0, sigma_e);
  arma::mat y_star = X * betas + u_hat + e_star + u_star;
  if (transformation == "log") {
    y_star = arma::exp(y_star);
  }
  y_star = arma::clamp(y_star, 0, arma::datum::inf);
  return List::create(Named("y_star") = y_star,
                      Named("u_i") = u_i,
                      Named("u_hat") = u_hat,
                      Named("e_star") = e_star,
                      Named("u_star") = u_star);
}


double col_means(arma::mat x){
  arma::mat X = arma::vec(x.begin(), x.n_rows, x.n_cols, false);
  return arma::mean(arma::mean(X));
}

double col_head_count(arma::mat x, double threshold){
  arma::mat X = arma::mat(x.begin(), x.n_rows, x.n_cols, false);
  arma::umat X_lt = X < threshold;
  arma::mat X_out = arma::conv_to<arma::mat>::from(X_lt);
  return arma::mean(arma::mean(X_out));
}

arma::mat area_means(arma::mat X, arma::uvec levels, arma::uvec T, double threshold) {
  int q = levels.n_rows;
  arma::mat out(q, 3);
  for (int i(0); i < q; i++) {
    int level = levels(i);
    arma::mat sub = X.rows(find(T == level));
    double colmeans = col_means(sub);
    double col_hc = col_head_count(sub, threshold);
    out(i, 0) = level;
    out(i, 1) = colmeans;
    out(i, 2) = col_hc;
  }
  return out;
}

// [[Rcpp::export]]
List mc_ebp(const arma::mat& X,
            const arma::vec& betas,
            const arma::vec& gamma,
            const arma::vec& u_hat,
            const IntegerVector& area_id,
            const int L,
            const double sigma_e,
            const double sigma_u,
            const double threshold,
            const std::string transformation) {

  arma::mat y_star_sum = arma::zeros(X.n_rows, L);
  // arma::mat mc_id = arma::zeros(X.n_rows, L);
  // arma::vec id(area_id.n_rows * L);

  CharacterVector levels_factor = area_id.attr("levels");
  arma::ivec area_id_vec = area_id;
  arma::uvec area_id_uvec = arma::conv_to<arma::uvec>::from(area_id_vec);
  arma::uvec levels = arma::unique(area_id_uvec);
  // int q = levels.n_rows;

  for (int i = 0; i < L; i++) {
    y_star_sum.col(i) = mc_census(X, betas, gamma, u_hat, area_id_uvec, L, sigma_e, sigma_u, transformation);
    // mc_id.col(i).fill(i+1);
    // if (i == 0) {
    //   id.subvec(0, area_id.n_rows-1) = area_id;
    // } else {
    //   id.subvec(i * area_id.n_rows, (i+1) * area_id.n_rows-1) = area_id;
    // }
  }

  // arma::vec y_star = vectorise(y_star_sum);
  // arma::vec mc_id_vec = vectorise(mc_id);
  // arma::mat y_star_final = arma::zeros(y_star.n_rows, 2);
  // y_star_final.col(0) = y_star;
  // y_star_final.col(1) = id;

  arma::mat Y_means = area_means(y_star_sum, levels, area_id_uvec, threshold);

  IntegerVector Area = wrap(Y_means.col(0));
  Area.attr("levels") = levels_factor;
  Area.attr("class") = "factor";

  DataFrame result = DataFrame::create(Named("Area") = Area,
                                       Named("Mean") = Y_means.col(1),
                                       Named("Head_Count") = Y_means.col(2));

  return List::create(Named("Mean") = result,
                      Named("Censuses_matrix") = y_star_sum,
                      Named("Area_id") = area_id);
}
