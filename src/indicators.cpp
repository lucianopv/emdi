#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include <algorithm>
#include <numeric>
#include <cmath>

// ---------------------------------------------------------------------------
// Indicator mask bits (for selective computation)
// ---------------------------------------------------------------------------
// Bit 0: Mean           Bit 1: HCR            Bit 2: PGap
// Bit 3: Gini           Bit 4: QSR
// Bit 5: Q10  Bit 6: Q25  Bit 7: Q50  Bit 8: Q75  Bit 9: Q90
//
// MASK_ALL = 0x3FF (all 10 bits set)
// MASK_NO_SORT = 0x007 (Mean + HCR + PGap only — no sorting needed)
static const int MASK_ALL     = 0x3FF;
static const int MASK_MEAN    = 0x001;
static const int MASK_HCR     = 0x002;
static const int MASK_PGAP    = 0x004;
static const int MASK_GINI    = 0x008;
static const int MASK_QSR     = 0x010;
static const int MASK_Q10     = 0x020;
static const int MASK_Q25     = 0x040;
static const int MASK_Q50     = 0x080;
static const int MASK_Q75     = 0x100;
static const int MASK_Q90     = 0x200;
static const int MASK_QUANTS  = 0x3E0; // all 5 quantiles
static const int MASK_NEEDS_SORT = 0x3F8; // Gini + QSR + all quantiles

// ---------------------------------------------------------------------------
// Helper: nth_element-based quantile (O(n) per quantile)
// For unweighted case: R's type=7 quantile
// ---------------------------------------------------------------------------
static double nth_element_quantile(std::vector<double>& x, double p) {
  int n = (int)x.size();
  if (n == 1) return x[0];
  double index = p * (n - 1);
  int lo = (int)std::floor(index);
  int hi = (int)std::ceil(index);
  if (lo == hi) {
    std::nth_element(x.begin(), x.begin() + lo, x.end());
    return x[lo];
  }
  // Need both lo and hi values
  std::nth_element(x.begin(), x.begin() + lo, x.end());
  double val_lo = x[lo];
  // After nth_element, elements after lo are >= x[lo]
  // Find min of elements from lo+1 onwards for hi
  double val_hi = *std::min_element(x.begin() + lo + 1, x.end());
  double frac = index - lo;
  return val_lo + frac * (val_hi - val_lo);
}

// ---------------------------------------------------------------------------
// Helper: weighted quantile matching R's wtd.quantile()
// Uses pre-sorted data
// ---------------------------------------------------------------------------
static double wtd_quantile_single(const std::vector<double>& x_sorted,
                                   const std::vector<double>& rw,
                                   double p) {
  int n = (int)x_sorted.size();
  if (p == 0.0) return x_sorted[0];
  if (p == 1.0) return x_sorted[n - 1];

  int sel = -1;
  for (int i = 0; i < n; ++i) {
    if (rw[i] >= p) { sel = i; break; }
  }
  if (sel == -1) return x_sorted[n - 1];

  if (rw[sel] == p && sel + 1 < n) {
    return (x_sorted[sel] + x_sorted[sel + 1]) / 2.0;
  }
  return x_sorted[sel];
}

// ---------------------------------------------------------------------------
// Helper: unweighted step quantile matching R's wtd.quantile()
// (R/framework_direct.R). With unit weights the cumulative weight fraction is
// rw[i] = (i + 1) / n, so the selection predicate rw[i] >= p is monotone in i
// and can be located by binary search -- no rw vector has to be materialised.
// NOTE: this is the inverse-CDF ("step") rule, deliberately NOT the type-7
// interpolation used for Quantile_10..Quantile_90. R's qsr() calls
// wtd.quantile() regardless of weights, while its quants() special-cases
// unit weights to stats::quantile(); the two must stay distinct here.
// Uses pre-sorted data.
// ---------------------------------------------------------------------------
static double unwtd_step_quantile(const std::vector<double>& x_sorted, double p) {
  int n = (int)x_sorted.size();
  if (p == 0.0) return x_sorted[0];
  if (p == 1.0) return x_sorted[n - 1];

  int lo = 0, hi = n - 1, sel = -1;
  while (lo <= hi) {
    int mid = lo + (hi - lo) / 2;
    if ((double)(mid + 1) / (double)n >= p) { sel = mid; hi = mid - 1; }
    else lo = mid + 1;
  }
  if (sel == -1) return x_sorted[n - 1];

  if ((double)(sel + 1) / (double)n == p && sel + 1 < n) {
    return (x_sorted[sel] + x_sorted[sel + 1]) / 2.0;
  }
  return x_sorted[sel];
}

// Sort x and weights together, compute cumulative weight fractions
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

// Check if all weights are exactly 1
static bool all_weights_one(const arma::vec& w) {
  for (arma::uword i = 0; i < w.n_elem; ++i)
    if (w[i] != 1.0) return false;
  return true;
}

// ---------------------------------------------------------------------------
// Core computation for a single domain with selective indicator mask.
//
// When mask omits Gini/QSR/quantiles, sorting is skipped entirely (O(n)).
// When Gini is not needed but quantiles are, uses nth_element (O(n) per quantile)
// instead of full sort (O(n log n)).
//
// Returns arma::vec of length 10 (unrequested indicators set to 0).
// ---------------------------------------------------------------------------
static arma::vec compute_domain_indicators_masked(const arma::vec& y,
                                                    const arma::vec& w,
                                                    double threshold,
                                                    int mask) {
  arma::vec result(10, arma::fill::zeros);
  int n = (int)y.n_elem;
  double sw = arma::sum(w);

  // --- Mean (O(n), no sort) ---
  if (mask & MASK_MEAN) {
    result[0] = arma::dot(y, w) / sw;
  }

  // --- HCR (O(n), no sort) ---
  if (mask & MASK_HCR) {
    double num = 0.0;
    for (int i = 0; i < n; ++i)
      if (y[i] < threshold) num += w[i];
    result[1] = num / sw;
  }

  // --- Poverty Gap (O(n), no sort) ---
  if (mask & MASK_PGAP) {
    double pgap = 0.0;
    for (int i = 0; i < n; ++i)
      if (y[i] < threshold)
        pgap += (1.0 - y[i] / threshold) * w[i];
    result[2] = pgap / sw;
  }

  // Early exit if no sort-dependent indicators needed
  if (!(mask & MASK_NEEDS_SORT)) return result;

  // Determine if we need a full sort (Gini or weighted quantiles) or partial
  bool need_gini = (mask & MASK_GINI) != 0;
  bool need_qsr = (mask & MASK_QSR) != 0;
  bool need_quants = (mask & MASK_QUANTS) != 0;
  bool is_unweighted = all_weights_one(w);

  // If Gini is needed OR weighted quantiles are needed, do full sort once
  bool need_full_sort = need_gini || ((need_qsr || need_quants) && !is_unweighted);

  std::vector<double> xs, ws_sorted, rw;
  std::vector<int> sort_idx;

  if (need_full_sort) {
    // Full sort (O(n log n))
    sort_idx.resize(n);
    std::iota(sort_idx.begin(), sort_idx.end(), 0);
    std::sort(sort_idx.begin(), sort_idx.end(),
              [&](int a, int b){ return y[a] < y[b]; });

    xs.resize(n); ws_sorted.resize(n);
    for (int i = 0; i < n; ++i) {
      xs[i] = y[sort_idx[i]];
      ws_sorted[i] = w[sort_idx[i]];
    }

    // Compute cumulative weight fractions for weighted quantiles
    if (!is_unweighted) {
      rw.resize(n);
      double cum = 0.0;
      for (int i = 0; i < n; ++i) { cum += ws_sorted[i]; rw[i] = cum / sw; }
    }
  }

  // --- Gini (needs full sort) ---
  if (need_gini) {
    std::vector<double> yw(n);
    for (int i = 0; i < n; ++i) yw[i] = xs[i] * ws_sorted[i];

    double cum_yw = 0.0, auc = 0.0;
    for (int i = 0; i < n; ++i) {
      auc += (cum_yw + yw[i] / 2.0) * ws_sorted[i];
      cum_yw += yw[i];
    }
    double sum_yw = cum_yw;
    auc = (auc / sw) / sum_yw;
    result[3] = 1.0 - 2.0 * auc;
  }

  // --- QSR (needs quantiles at 0.2 and 0.8) ---
  if (need_qsr) {
    double q20, q80;
    if (need_full_sort && !is_unweighted) {
      // Use sorted data
      q20 = wtd_quantile_single(xs, rw, 0.2);
      q80 = wtd_quantile_single(xs, rw, 0.8);
    } else if (need_full_sort && is_unweighted) {
      // Sorted, unweighted. R's qsr() uses wtd.quantile() (the step rule) even
      // with unit weights, so the type-7 interpolation used for the Quantile_*
      // indicators must NOT be applied here -- it shifts the quintile cut and
      // changes which observations fall in the bottom/top quintile.
      q20 = unwtd_step_quantile(xs, 0.2);
      q80 = unwtd_step_quantile(xs, 0.8);
    } else {
      // No full sort — use the weighted quantile with its own sort. Reached
      // when QSR is requested without Gini; note this covers the UNWEIGHTED
      // case too, where rw[i] = (i + 1) / n makes wtd_quantile_single() give
      // the same step rule as unwtd_step_quantile() above.
      sort_by_x(y, w, xs, ws_sorted, rw);
      q20 = wtd_quantile_single(xs, rw, 0.2);
      q80 = wtd_quantile_single(xs, rw, 0.8);
    }

    double sum_iq1_wy = 0.0, sum_iq1_w = 0.0;
    double sum_iq4_wy = 0.0, sum_iq4_w = 0.0;
    for (int i = 0; i < n; ++i) {
      if (y[i] <= q20) { sum_iq1_wy += w[i] * y[i]; sum_iq1_w += w[i]; }
      if (y[i] > q80)  { sum_iq4_wy += w[i] * y[i]; sum_iq4_w += w[i]; }
    }
    double top_mean = (sum_iq4_w > 0) ? sum_iq4_wy / sum_iq4_w : 0.0;
    double bot_mean = (sum_iq1_w > 0) ? sum_iq1_wy / sum_iq1_w : 1.0;
    result[4] = top_mean / bot_mean;
  }

  // --- Quantiles Q10, Q25, Q50, Q75, Q90 ---
  if (need_quants) {
    std::vector<double> probs = {0.10, 0.25, 0.50, 0.75, 0.90};
    std::vector<double> q(5);

    if (is_unweighted && !need_full_sort) {
      // nth_element approach — O(n) per quantile, no full sort
      std::vector<double> y_copy(n);
      for (int i = 0; i < n; ++i) y_copy[i] = y[i];
      for (int j = 0; j < 5; ++j) {
        // Make a fresh copy each time (nth_element is destructive)
        std::vector<double> yc(y_copy.begin(), y_copy.end());
        q[j] = nth_element_quantile(yc, probs[j]);
      }
    } else if (is_unweighted && need_full_sort) {
      // Already sorted — use type=7 directly
      for (int j = 0; j < 5; ++j) {
        double index = probs[j] * (n - 1);
        int lo = (int)std::floor(index), hi = (int)std::ceil(index);
        q[j] = xs[lo] + (index - lo) * (xs[hi] - xs[lo]);
      }
    } else {
      // Weighted — use sorted data
      if (!need_full_sort) {
        // Need to sort for weighted quantiles
        sort_by_x(y, w, xs, ws_sorted, rw);
      }
      for (int j = 0; j < 5; ++j) {
        q[j] = wtd_quantile_single(xs, rw, probs[j]);
      }
    }

    result[5] = q[0]; result[6] = q[1]; result[7] = q[2];
    result[8] = q[3]; result[9] = q[4];
  }

  return result;
}

// ---------------------------------------------------------------------------
// Exported: compute all 10 indicators for a single domain (backward-compatible)
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
arma::vec compute_domain_indicators_cpp(const arma::vec& y,
                                         const arma::vec& weights,
                                         double threshold) {
  return compute_domain_indicators_masked(y, weights, threshold, MASK_ALL);
}

// ---------------------------------------------------------------------------
// Exported: compute selected indicators for a single domain
// indicator_mask: bitmask selecting which indicators to compute
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
arma::vec compute_domain_indicators_selective_cpp(const arma::vec& y,
                                                    const arma::vec& weights,
                                                    double threshold,
                                                    int indicator_mask) {
  return compute_domain_indicators_masked(y, weights, threshold, indicator_mask);
}

// ---------------------------------------------------------------------------
// Exported: compute all indicators for all domains at once
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
arma::mat compute_all_indicators_cpp(const arma::vec& y,
                                      const arma::vec& weights,
                                      const arma::ivec& domain_ids,
                                      double threshold,
                                      int n_domains) {
  arma::mat result(n_domains, 10);

  int n = (int)y.n_elem;
  std::vector<int> starts(n_domains, -1), ends(n_domains, -1);

  for (int i = 0; i < n; ++i) {
    int d = domain_ids[i] - 1;
    if (starts[d] == -1) starts[d] = i;
    ends[d] = i;
  }

  for (int d = 0; d < n_domains; ++d) {
    if (starts[d] == -1) {
      for (int j = 0; j < 10; ++j)
        result(d, j) = arma::datum::nan;
      continue;
    }
    int s = starts[d], e = ends[d];
    arma::vec y_d = y.subvec(s, e);
    arma::vec w_d = weights.subvec(s, e);
    arma::vec indicators = compute_domain_indicators_masked(y_d, w_d, threshold, MASK_ALL);
    result.row(d) = indicators.t();
  }

  return result;
}
