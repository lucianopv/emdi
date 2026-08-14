# Freeze released emdi's ebp() indicators as a stored benchmark.
#
# Must be run while `emdi` still resolves to the RELEASED CRAN package, i.e.
# BEFORE this fork is renamed emdi2 -> emdi. Afterwards the name collides and
# the package would silently become its own oracle.

stopifnot(packageVersion("emdi") == "2.2.3")
cat("oracle package: emdi", as.character(packageVersion("emdi")),
    "at", dirname(system.file(package = "emdi")), "\n")

data("eusilcA_pop", package = "emdi")
data("eusilcA_smp", package = "emdi")

fixed <- eqIncome ~ gender + eqsize + cash + self_empl + unempl_ben +
  age_ben + surv_ben + sick_ben + dis_ben + rent + fam_allow +
  house_allow + cap_inv + tax_adj

# Identical argument list to test_qsr_cpp.R's end-to-end block. ebp() defaults
# to seed = 123, so this is reproducible without an explicit set.seed().
res <- emdi::ebp(
  fixed = fixed, pop_data = eusilcA_pop, pop_domains = "district",
  smp_data = eusilcA_smp, smp_domains = "district",
  threshold = 10859.24, transformation = "no", L = 20, MSE = FALSE
)

ind <- res$ind
cat("domains:", nrow(ind), " columns:", paste(names(ind), collapse = ", "), "\n")

out <- "tests/testthat/EBP/ebp_indicators_emdi223.csv"
# %.17g rather than write.csv()'s default 15 significant digits: 17 is the
# round-trip guarantee for an IEEE double, so the file reproduces the oracle
# bit-exactly rather than to ~1e-15 relative. A frozen benchmark should not
# quietly contribute error of its own to a 1e-8 comparison.
fmt <- ind
for (nm in names(fmt)) {
  if (is.numeric(fmt[[nm]])) fmt[[nm]] <- sprintf("%.17g", fmt[[nm]])
}
write.csv(fmt, out, row.names = FALSE, quote = FALSE)
cat("wrote", out, "\n")

# Round-trip check: the CSV must reproduce the in-memory oracle exactly, so a
# later parity test at 1e-8 is limited by the kernel, not by the file.
back <- read.csv(out, stringsAsFactors = FALSE)
num <- vapply(ind, is.numeric, logical(1))
stopifnot(identical(names(back), names(ind)), nrow(back) == nrow(ind))
worst <- max(abs(as.matrix(back[num]) - as.matrix(ind[num])), na.rm = TRUE)
cat("csv round-trip max abs diff:", format(worst, scientific = TRUE), "\n")
stopifnot(worst == 0)

cat("\nQuintile_Share head:\n")
print(utils::head(ind[, c("Domain", "Quintile_Share")], 4))
