
<!-- README.md is generated from README.Rmd. Please edit that file -->
grideR
======

The goal of **grideR** is to provides general infrastructure based on data.table for grid search in model hyperparameters space. Key feature: all preprocessing steps are performed within resamples. This package aims to be as easy to extend as possible.

Installation
------------

You can install dev version of **grideR** from [GitHub](https://github.com) with:

``` r
devtools::install_github("statist-bhfz/grideR")
```

Example
-------

``` r
library(resampleR)
#> Loading required package: data.table
library(grideR)
library(parallel)
        
# Input data
dt <- as.data.table(mtcars)
# data.table with resamples
splits <- resampleR::cv_base(dt, "hp")
# data.table with tunable model hyperparameters
xgb_grid <- CJ(
    max_depth = c(6, 8),
    eta = 0.025,
    colsample_bytree = 0.9,
    subsample = 0.8,
    gamma = 0,
    min_child_weight = c(3, 5),
    alpha = 0,
    lambda = 1
)
# Non-tunable parameters for xgboost
xgb_args <- list(
    nrounds = 500,
    early_stopping_rounds = 10,
    booster = "gbtree",
    eval_metric = "rmse",
    objective = "reg:linear",
    verbose = 0
)
# Dumb preprocessing function
# Real function will contain imputation, feature engineering etc.
# with all statistics computed on train folds and applied to validation fold
preproc_fun_example <- function(data) return(data[])

cl <- makePSOCKcluster(2L)
clusterExport(cl, list("dt", "splits", 
                       "xgb_grid", "xgb_args", 
                       "preproc_fun_example"))
clusterEvalQ(cl, {
    suppressMessages(library(resampleR))
    suppressMessages(library(grideR))
    suppressMessages(library(checkmate))
    suppressMessages(library(data.table))
    suppressMessages(library(xgboost))
})
#> [[1]]
#>  [1] "xgboost"    "checkmate"  "grideR"     "resampleR"  "data.table"
#>  [6] "stats"      "graphics"   "grDevices"  "utils"      "datasets"  
#> [11] "methods"    "base"      
#> 
#> [[2]]
#>  [1] "xgboost"    "checkmate"  "grideR"     "resampleR"  "data.table"
#>  [6] "stats"      "graphics"   "grDevices"  "utils"      "datasets"  
#> [11] "methods"    "base"

res <- clusterApply(
    cl, 
    splits, 
    function(split) across_grid(data = dt,
                                target = "hp",
                                split = split,
                                fit_fun = xgb_fit,
                                preproc_fun = preproc_fun_example,
                                grid = xgb_grid,
                                args = xgb_args,
                                metrics = c("rmse", "mae")))
res <- rbindlist(res, idcol = "split")
res
#>     split max_depth   eta colsample_bytree subsample gamma
#>  1:     1         6 0.025              0.9       0.8     0
#>  2:     1         6 0.025              0.9       0.8     0
#>  3:     1         8 0.025              0.9       0.8     0
#>  4:     1         8 0.025              0.9       0.8     0
#>  5:     2         6 0.025              0.9       0.8     0
#>  6:     2         6 0.025              0.9       0.8     0
#>  7:     2         8 0.025              0.9       0.8     0
#>  8:     2         8 0.025              0.9       0.8     0
#>  9:     3         6 0.025              0.9       0.8     0
#> 10:     3         6 0.025              0.9       0.8     0
#> 11:     3         8 0.025              0.9       0.8     0
#> 12:     3         8 0.025              0.9       0.8     0
#> 13:     4         6 0.025              0.9       0.8     0
#> 14:     4         6 0.025              0.9       0.8     0
#> 15:     4         8 0.025              0.9       0.8     0
#> 16:     4         8 0.025              0.9       0.8     0
#> 17:     5         6 0.025              0.9       0.8     0
#> 18:     5         6 0.025              0.9       0.8     0
#> 19:     5         8 0.025              0.9       0.8     0
#> 20:     5         8 0.025              0.9       0.8     0
#>     min_child_weight alpha lambda nrounds_best     rmse       mae
#>  1:                3     0      1          254 25.82795 23.442172
#>  2:                5     0      1          176 28.40845 24.490010
#>  3:                3     0      1          234 29.13382 26.868994
#>  4:                5     0      1          159 30.21812 26.535628
#>  5:                3     0      1          205 25.17265 20.216885
#>  6:                5     0      1          155 29.64170 25.587408
#>  7:                3     0      1          171 27.14264 20.503756
#>  8:                5     0      1          156 31.21857 27.003484
#>  9:                3     0      1          157 10.68297  9.809113
#> 10:                5     0      1          194 10.21284  8.511509
#> 11:                3     0      1          156 11.42183 10.083486
#> 12:                5     0      1          186 11.24998  9.353838
#> 13:                3     0      1          171 36.60020 27.660245
#> 14:                5     0      1          247 36.68939 25.642899
#> 15:                3     0      1          222 31.90733 27.354782
#> 16:                5     0      1          199 39.93131 27.828404
#> 17:                3     0      1          273 50.17284 31.402931
#> 18:                5     0      1          326 54.55717 32.347914
#> 19:                3     0      1          290 52.87341 30.706714
#> 20:                5     0      1          232 57.24948 33.237526

stopCluster(cl)
```
