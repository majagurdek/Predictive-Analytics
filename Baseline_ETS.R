df <- read.csv("clean_cycle_data.csv")
y <- ts(df$Total_Hires, start = c(2010, 8), frequency = 12)
library(forecast)

h <- 12

# RAW VERSION (no log transformation)

train <- head(y, -h)
test  <- tail(y, h)

sn_fc   <- snaive(train, h = h)
ets_fit <- ets(train)
ets_fc  <- forecast(ets_fit, h = h)

cat("\n========== RAW DATA ==========\n")
cat("\n=== SEASONAL NAIVE (raw) ===\n"); print(accuracy(sn_fc, test))
cat("\n=== ETS (raw) ===\n");            print(accuracy(ets_fc, test))
cat("\n=== ETS MODEL (raw) ===\n");      print(ets_fit)

# ============================================================
# LOG VERSION (with log transformation, back-transformed for accuracy)
# ============================================================
ly <- log(y)
train_log <- head(ly, -h)
test_log  <- tail(ly, h)
test_raw  <- tail(y, h)   # raw test set for back-transformed accuracy

sn_fc_log   <- snaive(train_log, h = h)
ets_fit_log <- ets(train_log)
ets_fc_log  <- forecast(ets_fit_log, h = h, biasadj = TRUE)

# Back-transform point forecasts to original scale
sn_fc_raw  <- exp(sn_fc_log$mean)
ets_fc_raw <- exp(ets_fc_log$mean)

cat("\n\n========== LOG-TRANSFORMED DATA ==========\n")
cat("\n=== SEASONAL NAIVE (log, back-transformed) ===\n")
print(accuracy(sn_fc_raw, test_raw))

cat("\n=== ETS (log, back-transformed) ===\n")
print(accuracy(ets_fc_raw, test_raw))

cat("\n=== ETS MODEL (log) ===\n")
print(ets_fit_log)

# ---- ETS residual diagnostics ----

# Log version (the one you discuss in the paper)
cat("\n=== ETS RESIDUAL DIAGNOSTICS (log) ===\n")
checkresiduals(ets_fit_log)        # prints Ljung-Box + plots
lb_ets_log <- checkresiduals(ets_fit_log, plot = FALSE)
print(lb_ets_log)                  # statistic, df, p-value on their own

# Raw version
cat("\n=== ETS RESIDUAL DIAGNOSTICS (raw) ===\n")
lb_ets_raw <- checkresiduals(ets_fit, plot = FALSE)
print(lb_ets_raw)
