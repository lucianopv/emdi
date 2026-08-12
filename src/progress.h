#ifndef EMDI_PROGRESS_H
#define EMDI_PROGRESS_H

#include <cmath>
#include <cstdio>
#include <ctime>
#include <string>

// C++ mirror of the progress formatting in R/progress.R.
//
// The parametric bootstrap loop lives entirely in C++ and cannot call back into
// R while it runs, so the format has to exist on both sides of the language
// boundary. tests/testthat/test_progress.R pins the two implementations to
// identical output so they cannot drift apart.
//
// Keep in sync with fmt_duration() and progress_line() in R/progress.R.

namespace emdi {

// HH:MM:SS, or "Nd HH:MM:SS" past 24 hours; unknown durations render as
// "--:--:--" rather than a number.
inline std::string fmt_duration(double secs) {
  if (!(secs >= 0.0) || !std::isfinite(secs)) return "--:--:--";
  long days = (long)std::floor(secs / 86400.0);
  long hh = (long)std::floor(std::fmod(secs, 86400.0) / 3600.0);
  long mm = (long)std::floor(std::fmod(secs, 3600.0) / 60.0);
  long ss = (long)std::floor(std::fmod(secs, 60.0));
  char buf[64];
  if (days > 0) {
    std::snprintf(buf, sizeof(buf), "%ldd %02ld:%02ld:%02ld", days, hh, mm, ss);
  } else {
    std::snprintf(buf, sizeof(buf), "%02ld:%02ld:%02ld", hh, mm, ss);
  }
  return std::string(buf);
}

// One progress line. `start_epoch` is seconds since the epoch, used only to
// project the finish clock time.
//
// std::localtime returns a pointer to a shared static buffer and is not
// thread-safe; every call site here is on the main thread, outside any OpenMP
// parallel region.
inline std::string progress_line(int i, int total, double elapsed,
                                 double start_epoch, const char* label) {
  int pct = (total > 0) ? (int)std::lround(100.0 * (double)i / (double)total) : 0;
  double remaining = (i <= 0)
    ? std::nan("")
    : (elapsed / (double)i) * (double)(total - i);

  char buf[256];
  std::snprintf(buf, sizeof(buf), "%s %d of %d (%d%%) | elapsed %s | remaining ~%s",
                label, i, total, pct,
                fmt_duration(elapsed).c_str(), fmt_duration(remaining).c_str());
  std::string out(buf);

  if (std::isfinite(remaining)) {
    std::time_t finish = (std::time_t)std::floor(start_epoch + elapsed + remaining);
    std::tm* lt = std::localtime(&finish);
    if (lt != NULL) {
      char tbuf[16];
      std::strftime(tbuf, sizeof(tbuf), "%H:%M:%S", lt);
      out += " | finish ~";
      out += tbuf;
    }
  }
  return out;
}

// The one-off line announcing what is about to run and when it started.
// Mirrors progress_header() in R/progress.R.
inline std::string progress_header(const char* title, int total,
                                   const char* label, double start_epoch) {
  std::time_t st = (std::time_t)std::floor(start_epoch);
  char tbuf[32] = "";
  std::tm* lt = std::localtime(&st);
  if (lt != NULL) std::strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", lt);

  char buf[256];
  std::snprintf(buf, sizeof(buf), "%s: %d %ss, started %s",
                title, total, label, tbuf);
  return std::string(buf);
}

}  // namespace emdi

#endif  // EMDI_PROGRESS_H
