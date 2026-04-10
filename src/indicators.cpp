#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include <algorithm>
#include <numeric>
#include <cmath>

// ---------------------------------------------------------------------------
// Helper: weighted quantile matching R's wtd.quantile()
//
// Algorithm:
//   1. Sort x (and weights) by x.
//   2. rw = cumsum(weights) / sum(weights)
//   3. For each prob p:
//        if p == 0 -> x[0]
//        if p == 1 -> x[n-1]
//        else select = first index where rw[select] >= p
//             if rw[select] == p  -> (x[select] + x[select+1]) / 2
//             else                -> x[select]
// ---------------------------------------------------------------------------
static double wtd_quantile_single(const std::vector<double>& x_sorted,
                                   const std::vector<double>& rw,
                                   double p) {
  int n = (int)x_sorted.size();
  if (p == 0.0) return x_sorted[0];
  if (p == 1.0) return x_sorted[n - 1];

  // first index where rw >= p
  int sel = -1;
  for (int i = 0; i < n; ++i) {
    if (rw[i] >= p) { sel = i; break; }
  }
  if (sel == -1) return x_sorted[n - 1]; // safety

  if (rw[sel] == p && sel + 1 < n) {
    return (x_sorted[sel] + x_sorted[sel + 1]) / 2.0;
  }
  return x_sorted[sel];
}

// Sort x and weights together, return sorted vectors + rw
static void sort_by_x(const arma::vec& x, const arma::vec& w,
                      std::vector<double>& xs, std::vector<double>& ws,
                      std::vector<double>& rw) {
  int n = (int)x.n_elem;
  std::vector<int> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](int a, int b){ return x[a] < x[b]; });

  xs.resize(n); ws.resize(n);
  for (int i = 0; i < n; ++i) { xs[i] = x[idx[i]]; ws[i] = w[idx[i]]; }

  double sw = 0.0;
  for (double wi : ws) sw += wi;

  rw.resize(n);
  double cum = 0.0;
  for (int i = 0; i < n; ++i) { cum += ws[i]; rw[i] = cum / sw; }
}

// Weighted quantile for a vector of probs
static std::vector<double> wtd_quantile(const arma::vec& x, const arma::vec& w,
                                         const std::vector<double>& probs) {
  std::vector<double> xs, ws, rw;
  sort_by_x(x, w, xs, ws, rw);

  std::vector<double> q(probs.size());
  for (size_t i = 0; i < probs.size(); ++i)
    q[i] = wtd_quantile_single(xs, rw, probs[i]);
  return q;
}

// ---------------------------------------------------------------------------
// Helper: unweighted quantile matching R's quantile(type=7)
// index = p*(n-1); interpolate linearly between floor and ceil elements
// ---------------------------------------------------------------------------
static double r_quantile_type7(const std::vector<double>& x_sorted, double p) {
  int n = (int)x_sorted.size();
  if (n == 1) return x_sorted[0];
  double index = p * (n - 1);
  int lo = (int)std::floor(index);
  int hi = (int)std::ceil(index);
  double frac = index - lo;
  return x_sorted[lo] + frac * (x_sorted[hi] - x_sorted[lo]);
}

static std::vector<double> unweighted_quantile(const arma::vec& x,
                                                const std::vector<double>& probs) {
  int n = (int)x.n_elem;
  std::vector<double> xs(n);
  for (int i = 0; i < n; ++i) xs[i] = x[i];
  std::sort(xs.begin(), xs.end());

  std::vector<double> q(probs.size());
  for (size_t i = 0; i < probs.size(); ++i)
    q[i] = r_quantile_type7(xs, probs[i]);
  return q;
}

// ---------------------------------------------------------------------------
// Helper: check if all weights are exactly 1 (unweighted case)
// ---------------------------------------------------------------------------
static bool all_weights_one(const arma::vec& w) {
  for (arma::uword i = 0; i < w.n_elem; ++i)
    if (w[i] != 1.0) return false;
  return true;
}

// ---------------------------------------------------------------------------
// Core computation for a single domain
//
// Returns arma::vec of length 10:
//   [0] Mean
//   [1] HCR
//   [2] PGap
//   [3] Gini
//   [4] QSR
//   [5] Q10
//   [6] Q25
//   [7] Q50
//   [8] Q75
//   [9] Q90
// ---------------------------------------------------------------------------
static arma::vec compute_domain_indicators(const arma::vec& y,
                                            const arma::vec& w,
                                            double threshold) {
  arma::vec result(10);
  int n = (int)y.n_elem;
  double sw = arma::sum(w);

  // --- Mean ---
  result[0] = arma::dot(y, w) / sw;

  // --- HCR ---
  {
    double num = 0.0;
    for (int i = 0; i < n; ++i)
      if (y[i] < threshold) num += w[i];
    result[1] = num / sw;
  }

  // --- Poverty Gap ---
  {
    double pgap = 0.0;
    for (int i = 0; i < n; ++i)
      if (y[i] < threshold)
        pgap += (1.0 - y[i] / threshold) * w[i];
    result[2] = pgap / sw;
  }

  // --- Gini ---
  // Matches R:
  //   pop_weights <- pop_weights[order(y)]
  //   y <- sort(y)
  //   auc <- sum((cumsum(c(0, (y * pop_weights)[1:(n-1)])) +
  //               ((y * pop_weights) / 2)) * pop_weights)
  //   auc <- (auc / sum(pop_weights)) / sum((y * pop_weights))
  //   G <- 1 - 2 * auc
  {
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return y[a] < y[b]; });

    std::vector<double> ys(n), ws_g(n), yw(n);
    for (int i = 0; i < n; ++i) {
      ys[i] = y[idx[i]];
      ws_g[i] = w[idx[i]];
      yw[i] = ys[i] * ws_g[i];
    }

    // cumsum(c(0, yw[0..n-2]))  has length n
    // element i = sum(yw[0..i-1])  (0-indexed), with element 0 = 0
    double cum_yw = 0.0;
    double auc = 0.0;
    for (int i = 0; i < n; ++i) {
      // cum_yw is cumsum(c(0,yw))[i]  = sum of yw[0..i-1]
      auc += (cum_yw + yw[i] / 2.0) * ws_g[i];
      cum_yw += yw[i];
    }

    double sum_yw = 0.0;
    for (int i = 0; i < n; ++i) sum_yw += yw[i];

    auc = (auc / sw) / sum_yw;
    result[3] = 1.0 - 2.0 * auc;
  }

  // --- QSR ---
  // Uses wtd.quantile at probs 0.2 and 0.8
  {
    std::vector<double> q20_80 = wtd_quantile(y, w, {0.2, 0.8});
    double q20 = q20_80[0], q80 = q20_80[1];

    double sum_iq1_wy = 0.0, sum_iq1_w = 0.0;
    double sum_iq4_wy = 0.0, sum_iq4_w = 0.0;
    for (int i = 0; i < n; ++i) {
      if (y[i] <= q20) { sum_iq1_wy += w[i] * y[i]; sum_iq1_w += w[i]; }
      if (y[i] > q80)  { sum_iq4_wy += w[i] * y[i]; sum_iq4_w += w[i]; }
    }
    result[4] = (sum_iq4_wy / sum_iq4_w) / (sum_iq1_wy / sum_iq1_w);
  }

  // --- Quantiles Q10, Q25, Q50, Q75, Q90 ---
  {
    std::vector<double> probs = {0.10, 0.25, 0.50, 0.75, 0.90};
    std::vector<double> q;

    if (all_weights_one(w)) {
      q = unweighted_quantile(y, probs);
    } else {
      q = wtd_quantile(y, w, probs);
    }

    result[5] = q[0];
    result[6] = q[1];
    result[7] = q[2];
    result[8] = q[3];
    result[9] = q[4];
  }

  return result;
}

// ---------------------------------------------------------------------------
// Exported: compute all 10 indicators for a single domain
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
arma::vec compute_domain_indicators_cpp(const arma::vec& y,
                                         const arma::vec& weights,
                                         double threshold) {
  return compute_domain_indicators(y, weights, threshold);
}

// ---------------------------------------------------------------------------
// Exported: compute all indicators for all domains at once
//
// domain_ids: integer vector 1..n_domains (contiguous blocks, sorted)
// Returns arma::mat [n_domains x 10]
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
arma::mat compute_all_indicators_cpp(const arma::vec& y,
                                      const arma::vec& weights,
                                      const arma::ivec& domain_ids,
                                      double threshold,
                                      int n_domains) {
  arma::mat result(n_domains, 10);

  // Find start/end indices for each domain (domains are contiguous 1..n_domains)
  int n = (int)y.n_elem;
  std::vector<int> starts(n_domains, -1), ends(n_domains, -1);

  int current_domain = -1;
  for (int i = 0; i < n; ++i) {
    int d = domain_ids[i] - 1; // 0-indexed
    if (starts[d] == -1) starts[d] = i;
    ends[d] = i;
  }

  for (int d = 0; d < n_domains; ++d) {
    if (starts[d] == -1) {
      // Empty domain: fill with NA
      for (int j = 0; j < 10; ++j)
        result(d, j) = arma::datum::nan;
      continue;
    }
    int s = starts[d], e = ends[d];
    arma::vec y_d = y.subvec(s, e);
    arma::vec w_d = weights.subvec(s, e);
    arma::vec indicators = compute_domain_indicators(y_d, w_d, threshold);
    result.row(d) = indicators.t();
  }

  return result;
}
