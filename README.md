
<!-- README.md is generated from README.Rmd. Please edit that file -->
tuneR
=====

The goal of **tuneR** is to provides general infrastructure based on data.table for grid search in model hyperparameters space. Key feature: all preprocessing steps are performed within resamples. This package aims to be as easy to extend as possible.

Installation
------------

You can install dev version of **tuneR** from [GitHub](https://github.com) with:

``` r
devtools::install_github("statist-bhfz/tuneR")
```

Example
-------

``` r
library(resampleR)
#> Loading required package: data.table
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
    suppressMessages(library(tuneR))
    suppressMessages(library(checkmate))
    suppressMessages(library(data.table))
    suppressMessages(library(xgboost))
})
#> [[1]]
#>  [1] "xgboost"    "checkmate"  "tuneR"      "resampleR"  "data.table"
#>  [6] "stats"      "graphics"   "grDevices"  "utils"      "datasets"  
#> [11] "methods"    "base"      
#> 
#> [[2]]
#>  [1] "xgboost"    "checkmate"  "tuneR"      "resampleR"  "data.table"
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
#>     min_child_weight alpha lambda nrounds_best     rmse      mae
#>  1:                3     0      1          149 14.07254 11.33070
#>  2:                5     0      1          295 14.94696 13.26584
#>  3:                3     0      1          155 13.85464 12.36905
#>  4:                5     0      1          212 19.00899 16.78825
#>  5:                3     0      1          237 15.28014 11.10874
#>  6:                5     0      1          138 18.90782 16.41260
#>  7:                3     0      1          187 15.15396 11.93570
#>  8:                5     0      1          134 20.26588 17.73880
#>  9:                3     0      1          116 21.94595 21.07602
#> 10:                5     0      1          127 21.18658 19.67070
#> 11:                3     0      1          131 20.68664 19.55153
#> 12:                5     0      1          127 22.32248 20.54899
#> 13:                3     0      1          406 58.25648 38.74395
#> 14:                5     0      1          325 49.76484 34.78882
#> 15:                3     0      1          393 57.34955 38.13544
#> 16:                5     0      1          258 51.77159 36.43976
#> 17:                3     0      1          117 17.68321 15.59404
#> 18:                5     0      1           88 21.81053 18.95525
#> 19:                3     0      1          107 20.44646 17.37061
#> 20:                5     0      1           93 19.53339 16.04594

stopCluster(cl)
```
