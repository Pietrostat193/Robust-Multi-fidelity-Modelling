# ============================================
# Data-fusion (HF monitors + LF satellite) for PM2.5
# Using CSV inputs:
#   - HF_clean_noNA.csv   (high-fidelity in-situ)
#   - LF_data.csv         (low-fidelity satellite / proxy)
# ============================================

# ---- 0) Packages -----------------------------------------------------------
need <- c("data.table","sf","dplyr","inlabru","fmesher")
has  <- need %in% rownames(installed.packages())
if (any(!has)) install.packages(need[!has], repos = "https://cloud.r-project.org")

if (!"INLA" %in% rownames(installed.packages())) {
  install.packages("INLA",
                   repos = c(getOption("repos"),
                             INLA = "https://inla.r-inla-download.org/R/stable"),
                   dep = TRUE)
}

library(sf)
library(dplyr)
library(inlabru)
library(fmesher)
library(INLA)
library(mgcv)
library(gbm)

# ---- 1) Read CSVs ----------------------------------------------------------
# Expect columns: site,time,x,y,value (any case). We'll standardize names.


HF_data <- ("HF_clean_noNA.csv", header = FALSE) 
colnames(HF_data)[1]="site"
colnames(HF_data)[2]="time"
colnames(HF_data)[3]="x"
colnames(HF_data)[4]="y"
colnames(HF_data)[5]="value"

# in-situ HF
LF_data <- read.csv("LF_data.csv") 
colnames(LF_data)[1]="site"
colnames(LF_data)[2]="time"
colnames(LF_data)[3]="x"
colnames(LF_data)[4]="y"
colnames(LF_data)[5]="value"
# satellite LF

# ---- 2) Keep LF rows from 15 nearest LF sites around each HF site ----------
# Site-level coordinates
# ---- 2) Keep LF rows from 15 nearest LF sites around each HF site ----------
# ---- 2) Keep LF rows from 15 nearest LF sites around each HF site ----------
HF_sites <- unique(HF_data$site)
LF_sites <- unique(LF_data$site)

HF_coords <- HF_data %>%
  distinct(site, .keep_all = TRUE) %>%
  arrange(match(site, HF_sites)) %>%
  select(x, y)

LF_coords <- LF_data %>%
  distinct(site, .keep_all = TRUE) %>%
  arrange(match(site, LF_sites)) %>%
  select(x, y)

XY <- as.matrix(rbind(HF_coords, LF_coords))
D  <- as.matrix(dist(XY))
nH <- nrow(HF_coords); nL <- nrow(LF_coords)
D  <- D[seq_len(nH), nH + seq_len(nL), drop = FALSE]

k <- min(15L, nL)
idx15 <- t(apply(D, 1, function(row) order(row)[seq_len(k)]))
LF_keep_sites <- sort(unique(LF_sites[as.vector(idx15)]))
LF_sub <- LF_data %>% filter(site %in% LF_keep_sites)

# ---- 3) Common times & windows of 30 --------------------------------------
common_time <- sort(intersect(unique(HF_data$time), unique(LF_sub$time)))
winSize <- 30L
nFull   <- floor(length(common_time) / winSize)
if (nFull == 0L) stop(sprintf("Not enough common time points for one %d-length window.", winSize))

# ---- 4) Helpers & output dir ----------------------------------------------
dir.create("models_r", showWarnings = FALSE)

rmse <- function(yhat, y) {
  idx <- is.finite(yhat) & is.finite(y)
  sqrt(mean((yhat[idx] - y[idx])^2))
}
mae <- function(yhat, y) {
  idx <- is.finite(yhat) & is.finite(y)
  mean(abs(yhat[idx] - y[idx]))
}

# version-agnostic like() constructor for inlabru
lk <- if (exists("bru_like", where = asNamespace("inlabru"), inherits = FALSE)) {
  inlabru::bru_like
} else {
  inlabru::like
}

alpha_grid <- c(0.8, 0.9, 1.0, 1.1, 1.2)

# Log now includes iteration index and new baseline metrics
log_df <- data.frame(
  iter   = seq_len(nFull),
  window = integer(nFull),
  alpha  = NA_real_,
  MAE_fusion  = NA_real_, RMSE_fusion  = NA_real_,
  MAE_gam     = NA_real_, RMSE_gam     = NA_real_,
  MAE_qgbrt   = NA_real_, RMSE_qgbrt   = NA_real_
)

set.seed(123)  # for GBM CV stability

# ---- 5) Window loop: fusion + LF-only baselines ----------------------------
for (w in seq_len(nFull)) {
  message(sprintf("=== Window %d/%d ===", w, nFull))
  idx_start  <- (w - 1L) * winSize + 1L
  idx_end    <- w * winSize
  keep_times <- common_time[idx_start:idx_end]
  
  HF_subtime <- HF_data %>% filter(time %in% keep_times)
  LF_subtime <- LF_sub   %>% filter(time %in% keep_times)
  
  HF_train <- HF_subtime %>% filter(site != 1L)
  HF_test  <- HF_subtime %>% filter(site == 1L)
  if (nrow(HF_train) == 0L || nrow(HF_test) == 0L) {
    warning(sprintf("Window %d: empty HF train or test; skipping.", w))
    next
  }
  
  # --------------------------- Fusion model (inlabru) -----------------------
  monitors <- HF_train %>%
    select(x, y, w = value) %>%
    st_as_sf(coords = c("x", "y"), crs = NA)
  
  proxy <- LF_subtime %>%
    select(x, y, sat = value) %>%
    st_as_sf(coords = c("x", "y"), crs = NA)
  
  pred_pts <- HF_test %>%
    select(x, y) %>%
    st_as_sf(coords = c("x", "y"), crs = NA)
  
  all_xy <- rbind(st_coordinates(monitors), st_coordinates(proxy))
  bound  <- fm_extensions(fm_nonconvex_hull(all_xy, convex = -0.05), c(0.1, 0.2))
  mesh   <- fm_mesh_2d(boundary = bound, max.edge = c(0.05, 0.3), cutoff = 0.02)
  
  matern <- inla.spde2.pcmatern(
    mesh, prior.range = c(0.5, 0.5), prior.sigma = c(1.0, 0.5), constr = FALSE
  )
  matern_err <- inla.spde2.pcmatern(
    mesh, prior.range = c(0.5, 0.5), prior.sigma = c(0.5, 0.5), constr = FALSE
  )
  
  cmp <- ~ -1 + beta0(1) + spde(main = geometry, model = matern) + err(main = geometry, model = matern_err)
  
  like_HF <- lk(
    family = "Gaussian",
    formula = w ~ beta0 + spde,
    data    = monitors,
    control.family = list(hyper = list(prec = list(prior = "pc.prec", param = c(1.0, 0.5))))
  )
  
  best_fit <- NULL; best_ml <- -Inf; best_a <- NA_real_
  for (a in alpha_grid) {
    like_LF <- lk(
      family = "Gaussian",
      formula = sat ~ I(a) * (beta0 + spde) + err,
      data    = proxy,
      include_latent = TRUE,
      control.family = list(hyper = list(prec = list(prior = "pc.prec", param = c(0.05, 0.5))))
    )
    
    fit <- try(
      inlabru::bru(
        cmp,
        like_HF, like_LF,
        options = list(control.inla = list(int.strategy = "eb"), bru_verbose = 0)
      ),
      silent = TRUE
    )
    if (inherits(fit, "try-error")) next
    mlik <- tryCatch(as.numeric(fit$mlik[1, 1]), error = function(e) NA_real_)
    if (!is.na(mlik) && mlik > best_ml) {
      best_ml  <- mlik
      best_fit <- fit
      best_a   <- a
    }
  }
  
  # Predict at HF test locations with fusion model
  if (!is.null(best_fit)) {
    preds_fusion <- predict(best_fit, newdata = pred_pts, formula = ~ beta0 + spde)
    yhat_fusion  <- as.numeric(preds_fusion$mean)
  } else {
    warning(sprintf("Window %d: fusion model fit failed.", w))
    yhat_fusion <- rep(NA_real_, nrow(HF_test))
  }
  
  # --------------------------- LF-only baselines ----------------------------
  # GAM (LF only; space+time smooths)
  # k controls smooth basis size; kept modest to avoid overfit in small windows
  gam_fit <- try(
    mgcv::gam(
      value ~ s(x, y, k = min(100, max(20, round(nrow(LF_subtime)/50)))) + s(time, k = min(10, length(unique(LF_subtime$time))-1)),
      data = LF_subtime,
      method = "REML"
    ),
    silent = TRUE
  )
  if (inherits(gam_fit, "try-error")) {
    yhat_gam <- rep(NA_real_, nrow(HF_test))
  } else {
    yhat_gam <- as.numeric(predict(gam_fit, newdata = HF_test))
  }
  
  # QGBRT (GBM) on LF only; predictors = x, y, time
  gbm_fit <- try(
    gbm::gbm(
      formula = value ~ x + y + time,
      data = LF_subtime,
      distribution = "gaussian",
      n.trees = 3000,
      interaction.depth = 5,
      shrinkage = 0.01,
      n.minobsinnode = 10,
      bag.fraction = 0.7,
      train.fraction = 1.0,
      cv.folds = 5,
      keep.data = FALSE,
      verbose = FALSE
    ),
    silent = TRUE
  )
  if (inherits(gbm_fit, "try-error")) {
    yhat_qgbrt <- rep(NA_real_, nrow(HF_test))
  } else {
    best_iter <- tryCatch(gbm::gbm.perf(gbm_fit, method = "cv", plot.it = FALSE),
                          error = function(e) which.max(gbm_fit$cv.error %>% replace(is.infinite(.), NA)))
    if (is.null(best_iter) || is.na(best_iter) || best_iter <= 0) best_iter <- gbm_fit$n.trees
    yhat_qgbrt <- as.numeric(predict(gbm_fit, newdata = HF_test, n.trees = best_iter))
  }
  
  # --------------------------- Metrics & logging ----------------------------
  ytrue <- HF_test$value
  
  log_df$window[w]     <- w
  log_df$alpha[w]      <- best_a
  log_df$MAE_fusion[w] <- mae(yhat_fusion, ytrue)
  log_df$RMSE_fusion[w] <- rmse(yhat_fusion, ytrue)
  
  log_df$MAE_gam[w]    <- mae(yhat_gam, ytrue)
  log_df$RMSE_gam[w]   <- rmse(yhat_gam, ytrue)
  
  log_df$MAE_qgbrt[w]  <- mae(yhat_qgbrt, ytrue)
  log_df$RMSE_qgbrt[w] <- rmse(yhat_qgbrt, ytrue)
  
  # Save fusion model + summaries (optional; LF baselines are lighter, usually not saved)
  if (!is.null(best_fit)) {
    summ <- list(
      mlik  = best_ml,
      alpha = best_a,
      fixed = tryCatch(best_fit$summary.fixed,    error = function(e) NULL),
      hyper = tryCatch(best_fit$summary.hyperpar, error = function(e) NULL)
    )
    saveRDS(list(fit = best_fit, summary = summ),
            file = sprintf("models_r/fusion_win%03d.rds", w))
  }
  
  # progress line
  message(sprintf("Progress %d/%d | Fusion: MAE=%.3f RMSE=%.3f | GAM: MAE=%.3f RMSE=%.3f | QGBRT: MAE=%.3f RMSE=%.3f",
                  w, nFull,
                  log_df$MAE_fusion[w], log_df$RMSE_fusion[w],
                  log_df$MAE_gam[w], log_df$RMSE_gam[w],
                  log_df$MAE_qgbrt[w], log_df$RMSE_qgbrt[w]))
}

# ---- 6) Save overall metrics ----------------------------------------------
write.csv(log_df, "models_r/window_metrics_with_baselines.csv", row.names = FALSE)
print(log_df)

cat("\nAverages across windows:\n",
    sprintf("Fusion   — MAE: %.3f | RMSE: %.3f\n",
            mean(log_df$MAE_fusion, na.rm = TRUE), mean(log_df$RMSE_fusion, na.rm = TRUE)),
    sprintf("GAM (LF) — MAE: %.3f | RMSE: %.3f\n",
            mean(log_df$MAE_gam,    na.rm = TRUE), mean(log_df$RMSE_gam,    na.rm = TRUE)),
    sprintf("QGBRT(LF) — MAE: %.3f | RMSE: %.3f\n",
            mean(log_df$MAE_qgbrt,  na.rm = TRUE), mean(log_df$RMSE_qgbrt,  na.rm = TRUE)),
    sep = "")
