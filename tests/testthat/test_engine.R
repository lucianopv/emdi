# One engine switch for the whole package.
#
# options(emdi.engine = c("auto", "cpp", "r")) gates BOTH the EBP and FH C++
# paths, at the level of an ebp()/fh() call rather than per kernel. A partial
# switch would be worse than none: a user setting "r" to debug a suspicious
# result would still get C++ underneath and conclude the C++ is fine.
#
# options(emdi.fh_engine) is kept as a deprecated alias so existing scripts and
# the FH test suite keep working.

test_that("emdi_engine defaults to auto, and auto means cpp", {
  withr::with_options(list(emdi.engine = NULL, emdi.fh_engine = NULL), {
    expect_identical(emdi_engine(), "auto")
    expect_true(.use_cpp())
  })
})

test_that("emdi_engine honours cpp and r", {
  withr::with_options(list(emdi.fh_engine = NULL), {
    withr::with_options(list(emdi.engine = "cpp"), {
      expect_identical(emdi_engine(), "cpp")
      expect_true(.use_cpp())
    })
    withr::with_options(list(emdi.engine = "r"), {
      expect_identical(emdi_engine(), "r")
      expect_false(.use_cpp())
    })
  })
})

test_that("an invalid value falls back to auto rather than erroring", {
  withr::with_options(list(emdi.fh_engine = NULL), {
    for (bad in list("nonsense", 42L, c("cpp", "r"), NA_character_)) {
      withr::with_options(list(emdi.engine = bad), {
        expect_identical(emdi_engine(), "auto")
      })
    }
  })
})

test_that("the deprecated emdi.fh_engine alias still works", {
  withr::with_options(list(emdi.engine = NULL), {
    withr::with_options(list(emdi.fh_engine = "r"), {
      expect_identical(emdi_engine(), "r")
      expect_false(.use_cpp())
    })
    withr::with_options(list(emdi.fh_engine = "cpp"), {
      expect_identical(emdi_engine(), "cpp")
    })
  })
})

test_that("the new option wins when both are set", {
  withr::with_options(list(emdi.engine = "r", emdi.fh_engine = "cpp"), {
    expect_identical(emdi_engine(), "r")
  })
})

test_that(".fh_use_cpp is retained and agrees with .use_cpp", {
  # FH call sites still reference .fh_use_cpp(); it must not diverge.
  for (e in c("auto", "cpp", "r")) {
    withr::with_options(list(emdi.engine = e, emdi.fh_engine = NULL), {
      expect_identical(.fh_use_cpp(), .use_cpp(), info = e)
    })
  }
})
