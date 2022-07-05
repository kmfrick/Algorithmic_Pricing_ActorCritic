# Multi-agent soft actor-critic in a competitive market
# Copyright (C) 2022 Kevin Michael Frick <kmfrick98@gmail.com>
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

library(stargazer)
library(zoo)

is.outlier <- function(series, Q1, Q3, IQR) {
  return(as.numeric(series < (Q1 - 1.5 * IQR) | series > (Q3 + 1.5 * IQR)))
}

print.multi.t.table <- function(data, caption, label, group.by, round = 3) {
  cat("\\begin{table}[t] \\centering\n")
  cat("\\caption{", caption, "}\n")
  cat("\\label{", label, "}\n")
  cat("\\begin{tabular}{@{\\extracolsep{5pt}}lcccc}\n")
  cat("\\\\[-1.8ex]\\hline\n")
  cat("\\hline \\\\[-1.8ex]\n")
  cat("Statistic & \\multicolumn{1}{c}{Mean} & \\multicolumn{1}{c}{Pctl(75)} & \\multicolumn{1}{c}{Median} & \\multicolumn{1}{c}{Pctl(25)} \\\\\n")
  for (gby in unique(data[[group.by]])) {
    cat("\\hline \\\\[-1.8ex]\n")
    cat("", group.by, " = ", gby , "\\\\\n")
    cat("\\hline \\\\[-1.8ex]\n")
    data.s <- data[data[[group.by]] == gby,]
    for (j in 1:ncol(data.s)) {
      if (names(data)[j] != group.by) {
        cat("", names(data.s)[j], " & ",
            round(mean(data.s[,j]), digits=round), " & ",
            round(quantile(data.s[,j], .75), digits=round), " & ",
            round(median(data.s[, j]), digits=round), " & ",
            round(quantile(data.s[, j], .25), digits=round), "\\\\\n")
      }
    }
  }
  cat("\\hline \\\\[-1.8ex]\n")
  cat("\\end{tabular}\n")
  cat("\\end{table}\n")
}

main <- function() {
  df <- read.csv("results/experiments_4agents.csv")
  suffix <- "."
  suffix.label <- "4ag"
  df <- df[,3:ncol(df)]
  print(mean(df$profit_gain))
  print(median(df$profit_gain))
  hist(df$profit_gain)
  return()

  # Compute outliers
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
  # Remove non-relevant columns and print table
  df.s <- subset(df, select=c(t, outlier_pg, outlier_dev_diff_prof, outlier_dev_disc_prof))
  names(df.s) <- c("t", "Outlier in profit gains", "Outlier in deviation profits (differential)", "Outlier in deviation profits (discounted)")
  print.multi.t.table(df.s, label=paste0("table:outliers", suffix.label), caption=paste0("Share of chaotic sessions during learning", suffix), group.by="t")

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
  for (dtype in levels(df$deviation_type)) {
    if (dtype == "br") { # Change to != to get other deviations only
      df.s <- df[df$deviation_type == dtype,]
      # Remove non-relevant columns and print table
      df.s <- subset(df.s, select=c(t, deviation_profit_percent, differential_deviation_profit, unprofitable_dev_diff, unprofitable_dev_disc))
      names(df.s) <- c("t", "Deviation gain (discounted, \\%)", "Deviation gain (differential)", "Unprofitable deviation (differential)", "Unprofitable deviation (discounted)")
      print.multi.t.table(df.s, label=paste0("table:dev", suffix.label), caption=paste0("Responses to deviations during learning", suffix), group.by = "t")
    }
  }
  df.s <- subset(df[df$t == 70000,], select = c(deviation_type, deviation_profit_percent, differential_deviation_profit, unprofitable_dev_diff, unprofitable_dev_disc))
  names(df.s) <- c("Deviation type", "Deviation gain (discounted, \\%)", "Deviation gain (differential)", "Unprofitable deviation (differiential)", "Unprofitable deviation (discounted)")
  df.s$`Deviation type` <- as.factor(df.s$`Deviation type`)
  levels(df.s$`Deviation type`) <- c("Static BR", "Monopoly", "Marginal cost", "Bertrand-Nash")
  print.multi.t.table(df.s, label=paste0("table:devtype", suffix.label), caption=paste0("Responses to various deviations", suffix), group.by = "Deviation type")
}

main()
