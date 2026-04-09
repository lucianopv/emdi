#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

arma::vec mc_census(arma::mat X, arma::vec betas, arma::vec e_star, arma::vec u_star, std::string transformation) {
  // Define the dimensions
  int n = X.n_rows;
  int p = X.n_cols;

  // arma::vec e_star = rnorm(n, 0, sigma_e);
  arma::vec y_star = X * betas + e_star + u_star;
  if (transformation == "log") {
    y_star = arma::exp(y_star);
  }
  y_star = arma::clamp(y_star, 0, arma::datum::inf);
  return y_star;
}

//[[Rcpp::export]]
double col_means(arma::vec x){
  arma::vec X = arma::vec(x.begin(), x.n_rows, false);
  return arma::mean(X);
}

//[[Rcpp::export]]
double col_head_count(arma::vec x, double threshold){
  arma::vec X = arma::vec(x.begin(), x.n_rows, false);
  arma::uvec X_lt = X < threshold;
  arma::vec X_out = arma::conv_to<arma::vec>::from(X_lt);
  return arma::mean(X_out);
}

//[[Rcpp::export]]
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
List mc_ebp_mse(const List& X,
                const List& theta,
                const List& Area,
                const int B,
                const int L,
                const Formula& formula,
                std::string transformation,
                const List& control,
                bool censuses) {

  DataFrame census = as<DataFrame>(X["census"]);
  DataFrame survey = as<DataFrame>(X["survey"]);
  arma::mat X_census = X["X_census"];
  arma::mat X_survey = X["X_survey"];
  // CharacterVector x_names = X["x_names"];
  std::string y_name = X["y_name"];

  DataFrame survey_df = clone(survey);

  int p = X_census.n_cols;
  int N = census.nrows();
  int n = survey.nrows();

  arma::vec betas = theta["betas"];
  // arma::vec gamma = theta["gamma"];
  // arma::vec u_hat = theta["u_hat"];
  double sigma_e = theta["sigma_e"];
  double sigma_u = theta["sigma_u"];
  double threshold = theta["threshold"];

  arma::uvec area = Area["area"];
  arma::uvec area_id = Area["area_id"];
  arma::uvec area_id_survey = Area["area_id_survey"];
  std::string id = as<std::string>(Area["id"]);

  // Create a dictionary of the names of the areas and the number of areas
  IntegerVector area_factor = census[id];
  IntegerVector area_unique = sort_unique(area_factor);
  CharacterVector area_factor_levels = area_factor.attr("levels");
  CharacterVector area_names_unique = unique(as<CharacterVector>(census[id]));
  int q = area_unique.length();
  std::map<int, std::string> area_names;
  for (int i = 0; i < q; i++) {
    area_names[area_unique(i)] = as<std::string>(area_factor_levels(i));
  }

  // X_census.insert_cols(0, arma::ones(N));
  // X_survey.insert_cols(0, arma::ones(n));

  // Create a list to store the results, which is of length B
  List results(B);

  // Access the mse from the list and estimate the overall mse for the mean and the head count over all the B simulations  per area
  arma::uvec levels = arma::unique(area_id);

  arma::mat mse_mean = arma::zeros(q, B);
  arma::mat mse_head_count = arma::zeros(q, B);
  arma::mat true_values_mean = arma::zeros(q, B);
  arma::mat true_values_head_count = arma::zeros(q, B);
  arma::mat ebp_estimates = arma::zeros(q, B);
  arma::mat ebp_head_count = arma::zeros(q, B);
  CharacterMatrix area_matrix = CharacterMatrix(q, B);
  arma::mat area_num_matrix = arma::zeros(q, B);
  arma::mat ebp_uhat = arma::zeros(q, B);
  arma::mat ebp_u_star = arma::zeros(N, B);
  arma::mat ebp_u_star_survey = arma::zeros(n, B);
  arma::mat ebp_e_star = arma::zeros(N, B);
  arma::mat ebp_e_star_survey = arma::zeros(n, B);
  arma::mat ebp_y_census = arma::zeros(N, B);
  arma::mat ebp_y_survey = arma::zeros(n, B);
  arma::mat ebp_mu_census = arma::zeros(N, B);
  arma::mat ebp_mu_survey = arma::zeros(n, B);


  Environment env = Environment::global_env();

  Function ebp_mc = env["ebp_mc"];

  for(int i = 0; i < B; i++) {

    // Create the random effects
    arma::vec e_star = rnorm(N, 0, sigma_e);
    arma::vec e_star_survey = rnorm(n, 0, sigma_e);
    arma::vec u_hat = rnorm(q, 0, sigma_u);
    ebp_uhat.col(i) = u_hat;

    // Create the vector of random effects for each observation
    arma::vec u_star = arma::zeros(N);
    arma::vec u_star_survey = arma::zeros(n);

    for (int j = 0; j < q; j++) {
      arma::uvec idx = find(area_id == levels(j));
      arma::uvec idx_survey = find(area_id_survey == levels(j));
      u_star.elem(idx) += u_hat(j);
      u_star_survey.elem(idx_survey) += u_hat(j);
    }
    ebp_u_star.col(i) = u_star;
    ebp_u_star_survey.col(i) = u_star_survey;


    arma::vec y_census(N);
    arma::vec y_survey(n);

    y_census = mc_census(X_census, betas, e_star, u_star, transformation);
    y_survey = mc_census(X_survey, betas, e_star_survey, u_star_survey, transformation);
    ebp_y_census.col(i) = y_census;
    ebp_y_survey.col(i) = y_survey;
    ebp_mu_census.col(i) = X_census * betas;
    ebp_mu_survey.col(i) = X_survey * betas;
    ebp_e_star.col(i) = e_star;
    ebp_e_star_survey.col(i) = e_star_survey;

    arma::mat true_values = area_means(y_census, levels, area_id, threshold);
    true_values_mean.col(i) = true_values.col(1);
    true_values_head_count.col(i) = true_values.col(2);

    survey_df[y_name] = y_survey;

    List ebp_b = ebp_mc(survey_df, census, formula, threshold, L, id, transformation, control, censuses);

    // Obtain the estimated values as a matrix and include the true values as well by comparing the area id
    arma::mat ind_mat = ebp_b["ind_matrix"];

    for (int j = 0; j < q; j++) {
      int level = true_values(j, 0);
      arma::uvec idx = find(ind_mat.col(0) == level);
      arma::vec ind_mat_mean = ind_mat.col(1);
      arma::vec ind_mat_head_count = ind_mat.col(2);
      mse_mean(j, i) = arma::as_scalar(arma::square(ind_mat_mean(idx) - true_values(j, 1)));
      mse_head_count(j, i) = arma::as_scalar(arma::square(ind_mat_head_count(idx) - true_values(j, 2)));
      ebp_estimates(j, i ) = arma::as_scalar(ind_mat_mean(idx));
      ebp_head_count(j, i) = arma::as_scalar(ind_mat_head_count(idx));
      area_matrix(j, i) = area_names[level];
      area_num_matrix(j, i) = level;
    }

    // DataFrame ind_b = as<DataFrame>(ebp_b["ind"]);

    // // Compare the true values with the estimated values
    // DataFrame mse = DataFrame::create(Named("Area") = ind_b[0],
    //                                   Named("Mean_test") = ind_b["Mean"],
    //                                   Named("Head_Count_test") = ind_b["Head_Count"],
    //                                   // Named("Area_id") = ind_b["area_id"],
    //                                   // Named("Area_true") = true_values.col(0),
    //                                   Named("Mean") = true_values.col(1),
    //                                   Named("Head_Count") = true_values.col(2),
    //                                   Named("MSE_Mean") = square(as<arma::vec>(ind_b["Mean"]) - true_values.col(1)),
    //                                   Named("MSE_Head_Count") = square(as<arma::vec>(ind_b["Head_Count"]) - true_values.col(2)),
    //                                   Named("B") = i);

    // mse["MSE_Mean"] = pow(as<NumericVector>(mse["Mean_test"]) - as<NumericVector>(mse["Mean"]), 2);
    // mse["MSE_Head_Count"] = pow(as<NumericVector>(mse["Head_Count_test"]) - as<NumericVector>(mse["Head_Count"]), 2);

    // mse_mean.col(i) = as<arma::vec>(mse["MSE_Mean"]);
    // mse_head_count.col(i) = as<arma::vec>(mse["MSE_Head_Count"]);

    // Save the results, the census, and the survey in a list
     results[i] = List::create( Named("ebp") = ebp_b,
                                Named("true_values") = true_values,
                                // Named("mse") = mse,
                                Named("survey") = survey_df);
                                // Named("census") = census);
                                // Named("ind") = ind);

  }

  List ebp = ebp_mc(survey, census, formula, threshold, L, id, transformation);
  DataFrame ind = as<DataFrame>(ebp["ind"]);

  arma::vec mse_mean_final = arma::mean(mse_mean, 1);
  arma::vec mse_head_count_final = arma::mean(mse_head_count, 1);

  // Create DataFrame to store the results of the mse for the mean and the head count over all the B simulations per area
  DataFrame mse_final = DataFrame::create(Named("Area") = ind[0],
                                          Named("Mean") = mse_mean_final,
                                          Named("Head_Count") = mse_head_count_final);


  // Create a list to store the censuses and the surveys from the B simulations
  // List censuses(B);
  // List surveys(B);

  // for (int i = 0; i < B; i++) {
  //   censuses[i] = results[i]["census"];
  //   surveys[i] = results[i]["survey"];
  // }

  List bootstrap_matrices = List::create(Named("u_hat") = ebp_uhat,
                                         Named("u_star") = ebp_u_star,
                                         Named("u_star_survey") = ebp_u_star_survey,
                                         Named("e_star") = ebp_e_star,
                                         Named("e_star_survey") = ebp_e_star_survey,
                                         Named("y_census") = ebp_y_census,
                                         Named("y_survey") = ebp_y_survey,
                                         Named("mu_census") = ebp_mu_census,
                                         Named("mu_survey") = ebp_mu_survey);

  // Return the results of the final mse, the mse for the mean and the head count over all the B simulations per area, the censuses, and the surveys
  return List::create(Named("mse") = mse_final,
                      Named("ind") = ind,
                      Named("area_matrix") = area_matrix,
                      Named("area_num_matrix") = area_num_matrix,
                      Named("area_id") = area_id,
                      // Named("area_names_unique") = area_names_unique,
                      Named("area_unique") = area_unique,
                      Named("area_factor_levels") = area_factor_levels,
                      Named("area_factor") = area_factor,
                      // Named("levels") = levels,
                      Named("Est_Mean") = ebp_estimates,
                      Named("Est_Head_Count") = ebp_head_count,
                      Named("Mean") = mse_mean,
                      Named("True_Mean") = true_values_mean,
                      Named("Head_Count") = mse_head_count,
                      Named("True_Head_Count") = true_values_head_count,
                      Named("bootstrap_matrices") = bootstrap_matrices,
                      Named("results") = results);
                                    // Named("censuses") = censuses,
                                    // Named("surveys") = surveys)

}
