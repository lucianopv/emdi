# Engine switch for the C++ Fay-Herriot fast path.
# Controlled by options(emdi.fh_engine = c("auto", "cpp", "r")); default "auto".
# "auto" and "cpp" use the C++ kernels where a fast path exists; "r" forces the
# legacy R implementation (oracle / fallback). Out-of-scope configs always run R
# regardless of this option.
.fh_engine <- function() {
  eng <- getOption("emdi.fh_engine", "auto")
  if (!is.character(eng) || length(eng) != 1L || !eng %in% c("auto", "cpp", "r")) {
    eng <- "auto"
  }
  eng
}

.fh_use_cpp <- function() {
  .fh_engine() %in% c("auto", "cpp")
}
