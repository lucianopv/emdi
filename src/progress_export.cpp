#include <Rcpp.h>
#include "progress.h"

// Thin exports so tests/testthat/test_progress.R can assert that the C++
// progress formatting matches R/progress.R exactly. The bootstrap loop uses
// emdi::progress_line() directly; these wrappers exist so the cross-language
// format contract is verifiable rather than assumed.

// [[Rcpp::export]]
std::string fmt_duration_cpp(double secs) {
  return emdi::fmt_duration(secs);
}

// [[Rcpp::export]]
std::string progress_line_cpp(int i, int total, double elapsed,
                              double start_epoch, std::string label) {
  return emdi::progress_line(i, total, elapsed, start_epoch, label.c_str());
}

// [[Rcpp::export]]
std::string progress_header_cpp(std::string title, int total,
                                std::string label, double start_epoch) {
  return emdi::progress_header(title.c_str(), total, label.c_str(), start_epoch);
}
