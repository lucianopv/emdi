# Validation of a user-supplied `true_indicators` frame.
#
# The old check was `all(!true_indicators$Domain %in% target)`, TRUE only when
# NO domain matched, after which the frame was consumed positionally via
# `as.matrix(true_indicators[, -1])`. Two silent-wrong-answer cases got through:
# a frame with the right domains in a DIFFERENT ORDER, and a frame with one
# wrong domain among many correct ones. Both produced MSE computed against the
# wrong domains' truths with no error at all. Column order was assumed too --
# the code assigned the canonical names onto whatever columns it was handed.

all_names <- c("Mean", "Head_Count", "Poverty_Gap", "Gini",
               "Quintile_Share", "Quantile_10", "Quantile_25",
               "Median", "Quantile_75", "Quantile_90")

# Minimal stand-in for the parts of `framework` the helper reads.
fake_fw <- function(domains, agg = NULL) {
  list(pop_domains_vec = factor(domains, levels = unique(domains)),
       aggregate_to_vec = if (is.null(agg)) NULL else
         factor(agg, levels = unique(agg)))
}

ti_frame <- function(domains, value_from = 1) {
  df <- data.frame(Domain = domains, stringsAsFactors = FALSE)
  for (i in seq_along(all_names)) {
    df[[all_names[i]]] <- value_from + seq_along(domains) + i * 100
  }
  df
}

test_that("a correctly specified frame is returned in output-domain order", {
  fw <- fake_fw(c("c", "a", "b"))
  ti <- ti_frame(c("c", "a", "b"))
  out <- normalise_true_indicators(ti, fw)

  expect_equal(dim(out), c(3L, 10L))
  expect_equal(colnames(out), all_names)
  expect_equal(as.numeric(out[, "Mean"]), ti$Mean)
})

test_that("rows supplied in a different order are reordered, not consumed positionally", {
  fw <- fake_fw(c("c", "a", "b"))          # output order: c, a, b
  shuffled <- ti_frame(c("a", "b", "c"))   # user supplied: a, b, c

  out <- normalise_true_indicators(shuffled, fw)

  # Row 1 must be domain "c"'s value, not the frame's first row ("a").
  expect_equal(as.numeric(out[1, "Mean"]), shuffled$Mean[shuffled$Domain == "c"])
  expect_equal(as.numeric(out[2, "Mean"]), shuffled$Mean[shuffled$Domain == "a"])
  expect_equal(as.numeric(out[3, "Mean"]), shuffled$Mean[shuffled$Domain == "b"])
  # The old code would have returned the frame's own order:
  expect_false(isTRUE(all.equal(as.numeric(out[, "Mean"]), shuffled$Mean)))
})

test_that("one wrong domain among many correct ones is rejected", {
  fw <- fake_fw(c("a", "b", "c"))
  ti <- ti_frame(c("a", "b", "ZZZ"))
  expect_error(normalise_true_indicators(ti, fw), "Unexpected")
  expect_error(normalise_true_indicators(ti, fw), "Missing")
})

test_that("missing, extra and duplicated domains are all rejected", {
  fw <- fake_fw(c("a", "b", "c"))
  expect_error(normalise_true_indicators(ti_frame(c("a", "b")), fw), "Missing")
  expect_error(normalise_true_indicators(ti_frame(c("a", "b", "c", "d")), fw),
               "Unexpected")
  expect_error(normalise_true_indicators(ti_frame(c("a", "b", "b")), fw),
               "duplicated")
})

test_that("columns are matched by name, not position", {
  fw <- fake_fw(c("a", "b"))
  ti <- ti_frame(c("a", "b"))
  reversed <- ti[, c("Domain", rev(all_names))]

  out <- normalise_true_indicators(reversed, fw)
  expect_equal(as.numeric(out[, "Mean"]), ti$Mean)
  expect_equal(as.numeric(out[, "Gini"]), ti$Gini)
})

test_that("only the MSE_indicators columns are required, others left at zero", {
  fw <- fake_fw(c("a", "b"))
  ti <- ti_frame(c("a", "b"))[, c("Domain", "Mean", "Head_Count")]

  out <- normalise_true_indicators(ti, fw,
                                   MSE_indicators = c("Mean", "Head_Count"))
  expect_equal(colnames(out), all_names)
  expect_equal(as.numeric(out[, "Mean"]), ti$Mean)
  # Unrequested slots are zero, matching the C++ kernel, which leaves
  # unmasked indicator positions at zero so the difference is zero either way.
  expect_true(all(out[, setdiff(all_names, c("Mean", "Head_Count"))] == 0))
})

test_that("a column required by MSE_indicators but absent is rejected", {
  fw <- fake_fw(c("a", "b"))
  ti <- ti_frame(c("a", "b"))[, c("Domain", "Mean")]
  expect_error(
    normalise_true_indicators(ti, fw, MSE_indicators = c("Mean", "Gini")),
    "Gini"
  )
})

test_that("domains are taken from aggregate_to when it is used", {
  # pop_domains has 4 values; aggregate_to collapses them to 2.
  fw <- fake_fw(domains = c("d1", "d2", "d3", "d4"),
                agg = c("R1", "R1", "R2", "R2"))
  # A frame at pop_domains level must be rejected...
  expect_error(normalise_true_indicators(ti_frame(c("d1", "d2", "d3", "d4")), fw),
               "aggregate_to")
  # ...and one at aggregate_to level accepted, in aggregate order.
  out <- normalise_true_indicators(ti_frame(c("R2", "R1")), fw)
  expect_equal(nrow(out), 2L)
  expect_equal(as.numeric(out[1, "Mean"]), ti_frame(c("R2", "R1"))$Mean[2])  # R1 is first
})

test_that("non-numeric and NA values are rejected", {
  fw <- fake_fw(c("a", "b"))
  bad <- ti_frame(c("a", "b")); bad$Mean <- c("x", "y")
  expect_error(normalise_true_indicators(bad, fw), "numeric")

  na_frame <- ti_frame(c("a", "b")); na_frame$Gini <- c(1, NA)
  expect_error(normalise_true_indicators(na_frame, fw), "NA")
})

test_that("a non-data-frame or a frame without Domain is rejected", {
  fw <- fake_fw(c("a", "b"))
  expect_error(normalise_true_indicators(matrix(1, 2, 11), fw), "data frame")
  no_dom <- ti_frame(c("a", "b")); no_dom$Domain <- NULL
  expect_error(normalise_true_indicators(no_dom, fw), "Domain")
})
