library(stargazer)
library(zoo)

df <- read.csv("experiments_new.csv")
df <- df[,3:ncol(df)]

# df$dev_profit_percent_coop <- NA
# df$dev_profit_percent_cost <- NA
# df$dev_profit_percent_nash <- NA
# df$dev_profit_percent_br <- NA
# df$dev_profit_diff_coop <- NA
# df$dev_profit_diff_cost <- NA
# df$dev_profit_diff_nash <- NA
# df$dev_profit_diff_br <- NA
# # Separate columns for each deviation type
# for (dtype in c("nash", "br", "coop", "cost")) {
#   for (seed in unique(df$seed)) {
#     for (t in unique(df$t)) {
#       df[df$seed == seed & df$t == t,paste0("dev_profit_percent_", dtype)] <- df[df$seed == seed &df$t == t & df$deviation_type == dtype,"deviation_profit_percent"]
#       df[df$seed == seed & df$t == t,paste0("dev_profit_diff_", dtype)] <- df[df$seed == seed & df$t == t & df$deviation_type == dtype,"differential_deviation_profit"]
#     }
#   }
# }
# df <- subset(df, select=-c(deviation_type, deviation_profit_percent, differential_deviation_profit))
# df <- unique(df)
# 
# 
# df$unprofitable_dev_diff_coop <- as.numeric(df$dev_profit_diff_coop< 0)
# df$unprofitable_dev_diff_cost <- as.numeric(df$dev_profit_diff_cost< 0)
# df$unprofitable_dev_diff_nash <- as.numeric(df$dev_profit_diff_nash< 0)
# df$unprofitable_dev_diff_br <- as.numeric(df$dev_profit_diff_br< 0)
# df$unprofitable_dev_percent_coop <- as.numeric(df$dev_profit_percent_coop< 0)
# df$unprofitable_dev_percent_cost <- as.numeric(df$dev_profit_percent_cost< 0)
# df$unprofitable_dev_percent_nash <- as.numeric(df$dev_profit_percent_nash< 0)
# df$unprofitable_dev_percent_br <- as.numeric(df$dev_profit_percent_br< 0)
df$unprofitable_dev_diff <- as.numeric(df$differential_deviation_profit < 0)
df$unprofitable_dev_disc <- as.numeric(df$deviation_profit_percent < 0)
df$deviation_type <- as.factor(df$deviation_type)
for (t in unique(df$t)) {
  for (dtype in levels(df$deviation_type)) {
    if (dtype == "br") { # Change to != to get other deviations only
    df.s <- df[df$t == t & df$deviation_type == dtype,]
    # Remove non-relevant columns and print table
    df.s <- subset(df.s, select=-c(seed, experiment_name, t, actor_hidden_size, discount, deviation_type, profit_gain))
    names(df.s) <- c("Deviation gain (discounted)", "Deviation gain (differential)", "Unprofitable deviation (differential)", "Unprofitable deviation (discounted)")
    stargazer(df.s, summary=TRUE, summary.stat = c("mean", "p75", "median", "p25"),style="aer", table.placement="t", label=paste0("table:desc2ag"), title=paste0("Descriptive statistics of experimental results for t = ", t - 3))
    }
  }
}

is.outlier <- function(series, Q1, Q3, IQR) {
  return(as.numeric(series < (Q1 - 1.5 * IQR) | series > (Q3 + 1.5 * IQR)))
}

Q1.pg <- quantile(df$profit_gain, .25)
Q3.pg <- quantile(df$profit_gain, .75)
IQR.pg <- IQR(df$profit_gain)
df$outlier_pg <- is.outlier(df$profit_gain, Q1.pg, Q3.pg, IQR.pg)
Q1.dev.disc.prof <- quantile(df$deviation_profit_percent, .25)
Q3.dev.disc.prof <- quantile(df$deviation_profit_percent, .75)
IQR.dev.disc.prof <- IQR(df$deviation_profit_percent)
Q1.dev.diff.prof <- quantile(df$differential_deviation_profit, .25)
Q3.dev.diff.prof <- quantile(df$differential_deviation_profit, .75)
IQR.dev.diff.prof <- IQR(df$differential_deviation_profit)
df$outlier_dev_diff_prof <- is.outlier(df$differential_deviation_profit, Q1.dev.diff.prof, Q3.dev.diff.prof, IQR.dev.diff.prof)
df$outlier_dev_disc_prof <- is.outlier(df$deviation_profit_percent, Q1.dev.disc.prof, Q3.dev.disc.prof, IQR.dev.disc.prof) 
for (t in unique(df$t)) {
  df.s <- df[df$t == t,]
  # Remove non-relevant columns and print table
  df.s <- subset(df.s, select=c(outlier_pg, outlier_dev_diff_prof, outlier_dev_disc_prof))
  names(df.s) <- c("Outlier in profit gains", "Outlier in deviation profits (differential)", "Outlier in deviation profits (discounted)")
  stargazer(df.s, summary=TRUE, summary.stat = c("mean", "p75", "median", "p25"),style="aer", table.placement="t", label=paste0("table:pgoutliers"), title="Share of chaotic sessions during learning")
}


