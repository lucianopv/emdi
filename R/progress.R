# Iteration progress reporting.
#
# Long-running MSE estimators (jackknife over m domains, bootstrap over B
# iterations) report where they are and when they expect to finish. The
# formatting is separated into pure functions so it can be tested without
# depending on wall-clock timing.

# Format a duration in seconds as HH:MM:SS, or "Nd HH:MM:SS" past 24 hours.
# Non-finite or negative input renders as unknown rather than NA, so an
# unestimable ETA does not print "NA" at the user.
fmt_duration <- function(secs) {
  if (length(secs) != 1L || is.na(secs) || !is.finite(secs) || secs < 0) {
    return("--:--:--")
  }
  secs <- as.numeric(secs)
  days <- floor(secs / 86400)
  hh <- floor((secs %% 86400) / 3600)
  mm <- floor((secs %% 3600) / 60)
  ss <- floor(secs %% 60)
  if (days > 0) {
    sprintf("%dd %02d:%02d:%02d", days, hh, mm, ss)
  } else {
    sprintf("%02d:%02d:%02d", hh, mm, ss)
  }
}

# Build one progress line. `elapsed` is seconds since `start_time`; both are
# passed in rather than read from the clock so this stays deterministic.
#
# The remaining-time estimate is the mean cost per completed iteration times
# the number left. That is appropriate here because jackknife domains and
# bootstrap replicates cost roughly the same each; it would be a poor estimator
# for a loop whose per-iteration cost trends.
progress_line <- function(i, total, elapsed, start_time, label = "iteration") {
  pct <- if (total > 0) round(100 * i / total) else 0
  remaining <- if (i <= 0) NA_real_ else (elapsed / i) * (total - i)

  out <- sprintf(
    "%s %d of %d (%d%%) | elapsed %s | remaining ~%s",
    label, i, total, pct, fmt_duration(elapsed), fmt_duration(remaining)
  )
  # Only project a finish time once there is something to extrapolate from.
  if (is.finite(remaining)) {
    finish <- start_time + as.numeric(elapsed) + as.numeric(remaining)
    out <- paste0(out, " | finish ~", format(finish, "%H:%M:%S"))
  }
  out
}

# The one-off line announcing what is about to run and when it started.
# Mirrored in src/progress.h for the C++ bootstrap loop; keep the two in sync.
progress_header <- function(title, total, label, start_time) {
  sprintf(
    "%s: %d %ss, started %s",
    title, total, label, format(start_time, "%Y-%m-%d %H:%M:%S")
  )
}

# TRUE when stderr is attached to a terminal, so overwriting with "\r" renders
# correctly. FALSE when output is redirected to a log file, where "\r" would
# collapse the whole run onto one unreadable line.
progress_is_terminal <- function() {
  isTRUE(interactive()) ||
    isTRUE(tryCatch(isatty(stderr()), error = function(e) FALSE))
}

# Create a progress reporter. Returns a function to call with the current
# iteration index; it handles throttling and formatting.
#
#   p <- progress_reporter(total = m, label = "domain", title = "Jackknife MSE")
#   for (i in seq_len(m)) { ...; p(i) }
#
# Intermediate ticks closer together than `min_interval` seconds are dropped,
# so a fast loop does not spend its time writing to stderr. The final tick is
# never dropped, so the line always ends in a completed state.
progress_reporter <- function(total, label = "iteration", title = NULL,
                              in_place = NULL, min_interval = 0.5) {
  if (is.null(in_place)) in_place <- progress_is_terminal()

  start <- Sys.time()
  last <- start

  if (!is.null(title)) {
    message(progress_header(title, total, label, start))
  }

  function(i) {
    now <- Sys.time()
    final <- i >= total
    if (!final &&
        as.numeric(difftime(now, last, units = "secs")) < min_interval) {
      return(invisible(NULL))
    }
    last <<- now

    elapsed <- as.numeric(difftime(now, start, units = "secs"))
    line <- progress_line(i, total, elapsed, start, label)

    if (in_place) {
      # Overwrite in place; break the line only when finished.
      message("\r", line, appendLF = final)
    } else {
      message(line)
    }
    if (.Platform$OS.type == "windows") flush.console()
    invisible(NULL)
  }
}
