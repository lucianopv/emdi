#' Number of CPU cores emdi2 may use
#'
#' Resolves how many CPU cores emdi2 is allowed to use for a computation.
#' Called internally by \code{\link{ebp}} and \code{\link{fh}}; exported so
#' that users can check what budget is in effect.
#'
#' The budget is resolved in this order, first match winning:
#' \enumerate{
#'   \item the \code{cpus} argument, if supplied;
#'   \item \code{getOption("emdi2.cores")};
#'   \item the \code{OMP_NUM_THREADS} environment variable;
#'   \item 1.
#' }
#' The result is then capped at 2 while \code{R CMD check} is running (which
#' sets \code{_R_CHECK_LIMIT_CORES_}), and at the number of cores the machine
#' reports.
#'
#' emdi2 uses at most one core unless asked otherwise. This keeps it safe
#' inside parallel pipelines (\code{crew}, \code{future}, \code{targets}),
#' where several R processes each taking every core is a common and severe
#' slowdown.
#'
#' @param cpus optional number of cores. \code{NULL} (default) resolves from
#'   the option and environment as described above.
#' @return A single positive integer.
#' @examples
#' emdi_cores()
#' emdi_cores(2)
#' @export
emdi_cores <- function(cpus = NULL) {
  n <- cpus
  if (is.null(n)) n <- getOption("emdi2.cores", NULL)
  if (is.null(n)) {
    env <- suppressWarnings(as.integer(Sys.getenv("OMP_NUM_THREADS", "")))
    if (length(env) == 1L && !is.na(env) && env >= 1L) n <- env
  }
  if (is.null(n)) n <- 1L

  n <- suppressWarnings(as.integer(n))
  if (length(n) != 1L || is.na(n) || n < 1L) n <- 1L

  # R CMD check limits packages to two cores; the check farm is shared.
  if (nzchar(Sys.getenv("_R_CHECK_LIMIT_CORES_"))) n <- min(n, 2L)

  detected <- suppressWarnings(as.integer(parallel::detectCores()))
  if (length(detected) != 1L || is.na(detected) || detected < 1L) detected <- 1L

  as.integer(min(n, detected))
}
