# --------------------------------------------------------
# Calibration + Histogram (Clean Version)
# X-axis limits: 0 to 1
# Right Y-axis: Labels and title removed
# --------------------------------------------------------
suppressPackageStartupMessages({
  library(ggplot2)
  library(caret)
  library(Hmisc)
})

# ==== 1) User settings ====
g_bins      <- 40           # Number of bins (quantile-based)
right_max   <- 200000       # Max value for right-side count axis (used only for scaling histogram height)
loess_color <- "black"
loess_alpha <- 0.12         # Thin ribbon
loess_span  <- 1.2
loess_level <- 0.95
hist_fill   <- "gray60"
hist_alpha  <- 0.60
hist_border <- "black"

# Fix x-axis range
x_min <- 0
x_max <- 1

# ==== 2) Compute calibration data ====
stopifnot(all(c("true", "XGB") %in% names(calib_df)))

calib_obj <- calibration(true ~ XGB, data = calib_df, class = "1", cuts = g_bins)
cal_df <- calib_obj$data

# Convert percent scale (0–100) to probability scale (0–1)
cal_df$midpoint <- cal_df$midpoint / 100
cal_df$Percent  <- cal_df$Percent  / 100

# Guard against negative values
cal_df$Percent <- pmax(cal_df$Percent, 0)

# ==== 3) Compute histogram data ====
# For distribution shape, equal-width binning is often preferred over cut2(quantile),
# but we keep the original logic here.
bins <- cut2(calib_df$XGB, g = g_bins)
cuts <- attr(bins, "cuts")
if (length(cuts) < 3) cuts <- pretty(range(calib_df$XGB, na.rm = TRUE), n = g_bins)

brks <- sort(unique(c(min(calib_df$XGB, na.rm = TRUE), cuts, max(calib_df$XGB, na.rm = TRUE))))
mid  <- (head(brks, -1) + tail(brks, -1)) / 2
bin_id <- cut(calib_df$XGB, breaks = brks, include.lowest = TRUE, right = TRUE)
counts <- as.integer(table(bin_id))

hist_df <- data.frame(
  xmin     = head(brks, -1),
  xmax     = tail(brks, -1),
  midpoint = mid,
  count    = counts
)

# [Important] Scale histogram height (even if the right axis is not displayed,
# the relative scaling still matters)
scale_factor <- 1 / right_max
hist_df$scaled <- pmin(hist_df$count * scale_factor, 1)

# ==== 4) LOESS fitting ====
fit <- loess(
  Percent ~ midpoint,
  data    = cal_df,
  span    = loess_span,
  control = loess.control(surface = "direct")
)

grid <- data.frame(midpoint = seq(0, 1, length.out = 400))
pr   <- predict(fit, newdata = grid, se = TRUE)
z <- qnorm(0.5 + loess_level / 2)

# Clip predictions and confidence intervals to [0, 1]
grid$fit <- pmin(pmax(pr$fit, 0), 1)
grid$lwr <- pmin(pmax(pr$fit - z * pr$se.fit, 0), 1)
grid$upr <- pmin(pmax(pr$fit + z * pr$se.fit, 0), 1)

# ==== 5) Plot ====
p <- ggplot() +
  # 1) Histogram (background)
  geom_rect(
    data = hist_df,
    aes(xmin = xmin, xmax = xmax, ymin = 0, ymax = scaled),
    fill = hist_fill, color = hist_border,
    alpha = hist_alpha, linewidth = 0.3
  ) +
  
  # 2) LOESS fit (ribbon and line)
  geom_ribbon(
    data = grid, aes(x = midpoint, ymin = lwr, ymax = upr),
    fill = loess_color, alpha = loess_alpha
  ) +
  geom_line(
    data = grid, aes(x = midpoint, y = fit),
    color = loess_color, linewidth = 1.0
  ) +
  
  # 3) Diagonal line (ideal calibration)
  geom_abline(
    slope = 1, intercept = 0,
    linetype = "dashed", color = "grey", linewidth = 1.0
  ) +
  
  # 4) Axis settings
  scale_x_continuous(
    name   = "Predicted Probability",
    limits = c(0, 1),          # Fix x-axis to [0, 1]
    expand = c(0, 0),          # Remove extra padding
    breaks = seq(0, 1, 0.2)
  ) +
  scale_y_continuous(
    name   = "Observed Proportion",
    limits = c(0, 1),
    expand = c(0, 0)
  ) +
  
  # 5) Theme settings
  labs(title = "XGB") +
  theme_minimal(base_size = 15) +
  theme(
    panel.grid = element_blank(),
    plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
    axis.title.x = element_text(size = 16, face = "bold", margin = margin(t = 10)),
    axis.title.y = element_text(size = 16, face = "bold", margin = margin(r = 10)),
    axis.text = element_text(size = 14),
    
    # Axis lines and ticks
    axis.line = element_line(color = "black", linewidth = 0.8),
    axis.ticks = element_line(color = "black", linewidth = 0.8),
    
    # Margins (minimize right margin)
    plot.margin = margin(10, 15, 10, 10)
  )

print(p)
