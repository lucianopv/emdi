# Engine switch for the C++ fast paths.
#
# Controlled by options(emdi.engine = c("auto", "cpp", "r")), default "auto".
# "auto" and "cpp" use the C++ kernels wherever a fast path exists; "r" forces
# the R implementation. Configurations with no C++ path always run R regardless.
#
# The switch is deliberately whole-call, not per kernel: `engine = "r"` means an
# ebp() or fh() call runs the R implementation end to end. A partial switch
# would be worse than none -- someone setting "r" to investigate a suspicious
# result would still get C++ underneath and wrongly conclude the C++ is fine.
#
# Its real purpose is verification. It lets anyone check the C++ against the R
# on their own data without reading any C++, which is what makes ~3,000 lines of
# kernels reviewable. The EBP side had no such switch, and that is precisely why
# the Quintile_Share quantile-rule bug (see NEWS) went unnoticed while every FH
# kernel was validated by flipping the engine and comparing.

#' Which computation engine emdi uses
#'
#' Reports whether emdi's C++ fast paths are in use. Set with
#' \code{options(emdi.engine = "cpp")} or \code{options(emdi.engine = "r")};
#' the default \code{"auto"} behaves as \code{"cpp"}.
#'
#' Forcing \code{"r"} runs the R implementation instead, which is useful for
#' checking a result against the reference code, or as a fallback if a C++
#' kernel misbehaves. Where no C++ path exists the R code runs either way.
#'
#' @return One of \code{"auto"}, \code{"cpp"} or \code{"r"}.
#' @examples
#' emdi_engine()
#' @export
emdi_engine <- function() {
  valid <- c("auto", "cpp", "r")

  eng <- getOption("emdi.engine", NULL)
  if (is.null(eng)) {
    # Deprecated alias, kept so existing scripts keep working.
    eng <- getOption("emdi.fh_engine", NULL)
  }
  if (is.null(eng)) return("auto")

  if (!is.character(eng) || length(eng) != 1L || is.na(eng) || !eng %in% valid) {
    return("auto")
  }
  eng
}

# TRUE when the C++ kernels should be used.
.use_cpp <- function() {
  emdi_engine() %in% c("auto", "cpp")
}

# Retained names: the FH call sites and their tests reference these. Defined in
# terms of the shared resolver so the two cannot drift apart.
.fh_engine <- function() emdi_engine()
.fh_use_cpp <- function() .use_cpp()
