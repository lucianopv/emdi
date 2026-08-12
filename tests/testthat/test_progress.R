# Progress reporting helpers (R/progress.R).
#
# The formatting is kept as pure functions taking an injected elapsed time and
# start time, so these tests do not depend on wall-clock timing.

test_that("fmt_duration renders seconds as HH:MM:SS and pads correctly", {
  expect_equal(fmt_duration(0), "00:00:00")
  expect_equal(fmt_duration(1), "00:00:01")
  expect_equal(fmt_duration(59), "00:00:59")
  expect_equal(fmt_duration(60), "00:01:00")
  expect_equal(fmt_duration(3599), "00:59:59")
  expect_equal(fmt_duration(3600), "01:00:00")
  expect_equal(fmt_duration(3661), "01:01:01")
})

test_that("fmt_duration switches to a day field beyond 24 hours", {
  expect_equal(fmt_duration(86400), "1d 00:00:00")
  expect_equal(fmt_duration(90061), "1d 01:01:01")
})

test_that("fmt_duration reports unknown durations rather than NA", {
  expect_equal(fmt_duration(NA_real_), "--:--:--")
  expect_equal(fmt_duration(Inf), "--:--:--")
})

test_that("progress_line reports position, percent, elapsed, remaining and finish", {
  start <- as.POSIXct("2026-08-12 14:03:12", tz = "UTC")
  # 37 of 94 done in 72s => 72/37*57 = 110.9s remaining => finish at 14:06:14
  line <- progress_line(
    i = 37, total = 94, elapsed = 72, start_time = start, label = "domain"
  )
  expect_match(line, "domain 37 of 94", fixed = TRUE)
  expect_match(line, "(39%)", fixed = TRUE)
  expect_match(line, "elapsed 00:01:12", fixed = TRUE)
  expect_match(line, "remaining ~00:01:50", fixed = TRUE)
  expect_match(line, "finish ~14:06:14", fixed = TRUE)
})

test_that("progress_line does not predict a finish time on the first tick", {
  start <- as.POSIXct("2026-08-12 14:03:12", tz = "UTC")
  line <- progress_line(
    i = 0, total = 94, elapsed = 0, start_time = start, label = "domain"
  )
  expect_match(line, "remaining ~--:--:--", fixed = TRUE)
  expect_no_match(line, "finish ~1", fixed = TRUE)
})

test_that("progress_reporter emits a header naming the total and start time", {
  msgs <- testthat::capture_messages({
    p <- progress_reporter(total = 94, label = "domain",
                           title = "Jackknife MSE", in_place = TRUE)
  })
  expect_length(msgs, 1)
  expect_match(msgs, "94", fixed = TRUE)
  expect_match(msgs, "started", fixed = TRUE)
  # A real clock time, not a placeholder.
  expect_match(msgs, "\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}")
})

test_that("progress_reporter in terminal mode overwrites in place and closes the line once", {
  p <- suppressMessages(progress_reporter(
    total = 3, label = "domain", title = "T", in_place = TRUE, min_interval = 0
  ))
  msgs <- testthat::capture_messages({ p(1); p(2); p(3) })

  expect_length(msgs, 3)
  expect_true(all(startsWith(msgs, "\r")))
  # Only the final tick terminates the line.
  expect_equal(sum(grepl("\n$", msgs)), 1)
  expect_true(grepl("\n$", msgs[3]))
})

test_that("progress_reporter in log mode emits whole lines and no carriage returns", {
  p <- suppressMessages(progress_reporter(
    total = 3, label = "domain", title = "T", in_place = FALSE, min_interval = 0
  ))
  msgs <- testthat::capture_messages({ p(1); p(2); p(3) })

  expect_length(msgs, 3)
  expect_false(any(grepl("\r", msgs, fixed = TRUE)))
  expect_equal(sum(grepl("\n$", msgs)), 3)
})

test_that("progress_reporter throttles intermediate ticks but never the last", {
  p <- suppressMessages(progress_reporter(
    total = 100, label = "domain", title = "T", in_place = TRUE,
    min_interval = 3600   # nothing intermediate can beat this
  ))
  msgs <- testthat::capture_messages({ for (i in 1:100) p(i) })

  # Only the final tick survives throttling.
  expect_length(msgs, 1)
  expect_match(msgs[1], "100 of 100", fixed = TRUE)
})

# --- cross-language format parity -------------------------------------------
# The C++ parametric bootstrap prints its own progress (it cannot call back into
# R from inside the loop), so the format necessarily exists twice. These tests
# pin the two implementations to the same output, so the duplication cannot
# drift apart silently.

test_that("C++ progress_line matches the R implementation character for character", {
  start <- as.POSIXct("2026-08-12 14:03:12")   # local tz, as C++ localtime uses
  cases <- list(
    c(i = 37, total = 94, elapsed = 72),
    c(i = 1,  total = 94, elapsed = 0.4),
    c(i = 94, total = 94, elapsed = 180),
    c(i = 0,  total = 94, elapsed = 0)         # no ETA yet
  )
  for (cs in cases) {
    expect_equal(
      progress_line_cpp(cs[["i"]], cs[["total"]], cs[["elapsed"]],
                        as.numeric(start), "bootstrap iteration"),
      progress_line(cs[["i"]], cs[["total"]], cs[["elapsed"]],
                    start, "bootstrap iteration"),
      info = paste("i =", cs[["i"]])
    )
  }
})

test_that("C++ fmt_duration matches the R implementation", {
  for (s in c(0, 1, 59, 60, 3599, 3600, 3661, 86400, 90061)) {
    expect_equal(fmt_duration_cpp(s), fmt_duration(s), info = paste("secs =", s))
  }
})

test_that("C++ progress_header matches the R implementation", {
  start <- as.POSIXct("2026-08-12 14:03:12")
  expect_equal(
    progress_header_cpp("Bootstrap MSE", 50L, "bootstrap iteration",
                        as.numeric(start)),
    progress_header("Bootstrap MSE", 50L, "bootstrap iteration", start)
  )
})
