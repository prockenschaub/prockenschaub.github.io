It is well known that prediction models have a tendency to overfit to
the training data, especially if we only have a limited amount of
training data. While performance of such overfit models appears high
when evaluated on the data available during training, their performance
on new, previously unseen data is often considerably worse. Although it
may be tempting to the analyst to choose a model with high training
performance, it is the model’s performance in future data that we really
interested in.

Several resampling methods have been proposed to account for this issue.
The most widely used techniques fall into two categories:
cross-validation and bootstrapping. The idea underlying these techniques
is similar. By repeating the model fitting multiple times on different
subsets of the training data, we may get a better understanding of the
magnitude of overfitting and can account for it in our model building
and evaluation. Without going into too much detail, cross-validation
separates the data into *k* mutually exclusive folds and always holds
one back as a “hidden” test set. Note that the sample size available to
the model during each training run necessarily decreases to
$\\frac{k-1}{k}n$. Bootstrap, on the other hand, resamples (with
replacement) a data set with the same size *n* as the original training
set and then — depending on the exact method — uses a weighted
combination randomly sampled and excluded observations.

Whereas the machine learning community almost exclusively uses
cross-validation for model validation, bootstrap-based methods may be
more commonly seen in biomedical sciences. One reason for this
popularity may be the fact that they are championed by preeminent
experts in the field: both Frank Harrell (Harrell 2015) and Ewout
Steyerberg (Steyerberg 2019) prominently feature the bootstrap — and in
particular the optimism-adjusted bootstrap (OAD) — in their textbooks.
In this post, I give a brief introduction into OAD and compare it to
repeated cross-validation and regular bootstrap. OAD is implemented in
the R packages [*caret*](https://cran.r-project.org/web/packages/caret/)
and Frank Harrell’s
[*rms*](https://cran.r-project.org/web/packages/rms/) but not in the
recent
[*tidymodels*](https://cran.r-project.org/web/packages/tidymodels/)
ecosystem (Kuhn and Silge 2022). This post will therefore provide a
step-by-step guide to doing OAD with
[*tidymodels*](https://cran.r-project.org/web/packages/tidymodels/).

Optimism-adjusted bootstrap
===========================

Like other resampling schemes, the OAD aims to avoid overly optimistic
estimation of model performance during internal validation — i.e.,
validation of model performance using the training dataset. As we will
see further down, simply calculating performance metrics on the same
data used for training leads to artificially high/good performance
estimates. We will call this the “apparent” performance. OAD proposes to
obtain a better estimate by directly estimating the amount of “optimism”
in the apparent performance. The steps needed to do so are as follows
(Steyerberg 2019):

1.  Fit a model *M* to the original training set *S* and use *M* to
    calculate the apparent performance *R*(*M*, *S*) (e.g., accuracy) on
    the training data
2.  Draw a bootstrapped sample *S*<sup>\*</sup> of the same size as *S*
    through sampling *with* replacement
3.  Construct another model *M*\* by performing all model building steps
    (pre-processing, imputation, model selection, etc.) on
    *S*<sup>\*</sup> and calculate it’s apparent performance
    *R*(*M*<sup>\*</sup>, *S*<sup>\*</sup>) on *S*\*
4.  Use *M*\* to estimate the performance *R*(*M*<sup>\*</sup>, *S*)
    that it would have had on the original data *S*.
5.  Calculate the optimism
    *O*<sup>\*</sup> = *R*(*M*<sup>\*</sup>, *S*<sup>\*</sup>) − *R*(*M*<sup>\*</sup>, *S*)
    as the difference between the apparent and test performance of
    *M*\*.
6.  Repeat steps 2.-5. may times *B* to obtain a sufficiently stable
    estimate (common recommendations range from 100-1000 times depending
    on the computational feasibility)
7.  Subtract the mean optimism $\\frac{1}{B} \\sum^B\_{b=1} O^\*\_b$
    from the apparent performance *R*<sub>*a**p**p*</sub> in the
    original training data *S* to get a optimism-adjusted estimate of
    model performance.

The basic intuition behind this procedure is that the model *M*\* will
overfit to *S*<sup>\*</sup> in the same way as *M* overfits to *S*. We
can then estimate the difference between *M*’s observed apparent
performance *R*(*M*, *S*) and its unobserved performance on future test
data *R*(*M*, *U*) from the difference between the bootstrapped model
*M*<sup>\*</sup>’s apparent performance
*R*(*M*<sup>\*</sup>, *S*<sup>\*</sup>) and its test performance
*R*(*M*<sup>\*</sup>, *S*) (which are both observed). The training data
*S* acts as a stand-in test data for the bootstrapped model *M*\*.

The following sections will apply this basic idea to the Ames housing
dataset and compare estimates derived via OAB to repeated
cross-validation and standard bootstrap.

The Ames data set
=================

The Ames data set contains information on 2,930 properties in Ames,
Iowa, and contains 74 variables including the number of bedrooms,
whether the property includes a garage, and the sale price. We choose
this data set because it provides a decent sample size for predictive
modelling and is already used prominently in the documentation of the R
`tidymodels` ecosystem. More information on the Ames data set can be
found in (Kuhn and Silge 2022).

``` r
set.seed(123)
library("tidyverse")
library("tidymodels")

data(ames)
dim(ames)
```

    ## [1] 2930   74

``` r
ames[1:5, 1:5]
```

    ## # A tibble: 5 x 5
    ##   MS_SubClass                         MS_Zoning                Lot_Frontage Lot_Area Street
    ##   <fct>                               <fct>                           <dbl>    <int> <fct> 
    ## 1 One_Story_1946_and_Newer_All_Styles Residential_Low_Density           141    31770 Pave  
    ## 2 One_Story_1946_and_Newer_All_Styles Residential_High_Density           80    11622 Pave  
    ## 3 One_Story_1946_and_Newer_All_Styles Residential_Low_Density            81    14267 Pave  
    ## 4 One_Story_1946_and_Newer_All_Styles Residential_Low_Density            93    11160 Pave  
    ## 5 Two_Story_1946_and_Newer            Residential_Low_Density            74    13830 Pave

For this exercise, we try to predict sale prices within the dataset. To
keep preprocessing simple, we limit the predictors to only numeric
variables, which we centre and scale. Since sale prices are right
skewed, we log them before prediction. Finally, we will hold back a
random quarter of the data to simulate external validation on an
independent identically distributed test set.

``` r
# Define sale price as the prediction target
formula <- Sale_Price ~ .

# Remove categorical variables, log sale price, scale the numeric predictors
preproc <- recipe(formula, data = ames[0, ]) %>% 
  step_rm(all_nominal_predictors()) %>% 
  step_log(all_outcomes()) %>% 
  step_normalize(all_numeric_predictors(), -all_outcomes())

# Randomly split into training (3/4) and testing (1/4) sets
train_test_split <- initial_split(ames, prop = 3/4)
train <- training(train_test_split)
test <- testing(train_test_split)
```

Optimism-adjusted bootstrap with *tidymodels*
=============================================

Now that we have set up the data, lets look into how we can build a
linear regression model and validate it via OAB. We proceed according to
the steps described above.

Step 1: Calculate apparent perforamnce
--------------------------------------

To start, we simply fit and evaulate our model *M* on the original
training data *S* (note that we also apply preprocessing, therefore we
strictly speaking train our model on the preprocessed data *S*′). Since
our outcome is a continuous value strictly greater than zero, we will
use the residual mean squared error as our performance metric.

``` r
prepped <- prep(preproc, train)
preproc_orig <- juice(prepped)
fit_orig <- fit(linear_reg(), formula, preproc_orig)
preds_orig <- predict(fit_orig, new_data = preproc_orig)
perf_orig <- rmse_vec(preproc_orig$Sale_Price, preds_orig$.pred)

perf_orig
```

    ## [1] 0.1693906

Step 2: Create bootstrapped samples
-----------------------------------

After obtaining *M* and *R*(*M*, *S*), we now produce a set of bootstrap
samples to estimate the amount of optimism in this performance estiamte.
We use the *tidymodels* sub-package *rsample* to create a data frame
`bs` with `200` bootstrap samples. All of these resamples have training
data of equal size to the original training data (n = 2197). Note
however that the “testing data” set aside differs between splits, as it
is defined by all rows that did not get sampled into the training data,
which is a random variable and may vary between bootstraps. We won’t use
this testing data for OAB but it is for example used in the simple
bootstrap that we use for comparison later.

``` r
bs <- bootstraps(train, times = 200)

bs %>% slice(1:5)
```

    ## # A tibble: 5 x 2
    ##   splits             id          
    ##   <list>             <chr>       
    ## 1 <split [2197/813]> Bootstrap001
    ## 2 <split [2197/818]> Bootstrap002
    ## 3 <split [2197/813]> Bootstrap003
    ## 4 <split [2197/786]> Bootstrap004
    ## 5 <split [2197/792]> Bootstrap005

``` r
bs %>% slice((n()-5):n())
```

    ## # A tibble: 6 x 2
    ##   splits             id          
    ##   <list>             <chr>       
    ## 1 <split [2197/825]> Bootstrap195
    ## 2 <split [2197/812]> Bootstrap196
    ## 3 <split [2197/800]> Bootstrap197
    ## 4 <split [2197/792]> Bootstrap198
    ## 5 <split [2197/805]> Bootstrap199
    ## 6 <split [2197/804]> Bootstrap200

Step 3: Fit bootstrapped models and calculate their apparent performance
------------------------------------------------------------------------

We now use the bootstrap data.frame `bs` to preprocess each sample
*S*<sup>\*</sup> individually, fit a linear regression *M*<sup>\*</sup>
to it, and calculate its apparent performance
*R*(*M*<sup>\*</sup>, *S*<sup>\*</sup>).

``` r
bs <- bs %>% 
  mutate(
    # Apply preprocessing separately for each bootstrapped sample S*
    processed = map(splits, ~ juice(prep(preproc, training(.)))),
    # Fit a separate model M* to each preprocessed bootstrap
    fitted = map(processed, ~ fit(linear_reg(), formula, data = .)),
    # Predict values for each bootstrap's training data S* and calculate RMSE
    pred_app = map2(fitted, processed, ~ predict(.x, new_data = .y)),
    perf_app = map2_dbl(processed, pred_app, ~ rmse_vec(.x$Sale_Price, .y$.pred))
  )
```

Step 4: Evaluate on the original training data
----------------------------------------------

Since we stored the fitted models *M*<sub>*i*</sub><sup>\*</sup> in a
column of the data.frame, we can easily re-use them to predict values
for the original data and evaluate them. Remember that because some of
the rows in the original dataset did not end up in the bootstrapped
dataset, we expect the performance
*R*(*M*<sub>*i*</sub><sup>\*</sup>, *S*) of each model
*M*<sub>*i*</sub><sup>\*</sup> to be lower than the performance in its
own training data
*R*(*M*<sub>*i*</sub><sup>\*</sup>, *S*<sub>*i*</sub><sup>\*</sup>).

``` r
bs <- bs %>% 
  mutate(
    pred_test = map(fitted, ~ predict(., new_data = preproc_orig)),
    perf_test = map_dbl(pred_test, ~ rmse_vec(preproc_orig$Sale_Price, .$.pred)),
  )
```

Step 5: Estimate the optimism
-----------------------------

The amount of optimism in our apparent estimate is now simply estimated
by the differences between apparent and test performance in each
bootstrap.

``` r
bs <- bs %>% 
  mutate(
    optim = perf_app - perf_test
  )
```

Steps 6-7: Adjust for optimism
------------------------------

We already repeated this procedure in parallel for 200 samples,
therefore step 6 is fulfilled. In order to get a single, final estimate,
all that’s left to do is to calculate the mean and standard deviation of
the optimism and substract them (which approximately normal 95% Wald
confidence limits) from the apparent performance obtained in step 1.
This is now the performance that we report for our model after internal
validation

``` r
mean_opt <- mean(bs$optim)
std_opt <- sd(bs$optim)

(perf_orig - mean_opt) + c(-2, 0, 2) * std_opt / sqrt(nrow(bs))
```

    ## [1] 0.1782025 0.1799199 0.1816372

External validation
-------------------

Remember that we set aside a quarter of the data for external validation
(external is a bit of misnomer here but more on that later). We can now
compare how our estimate from internal validation compares to the
performance in the held-out data. Indeed, the performance seems to have
slightly dropped but — thankfully — it is still within the bounds
suggested by OAB above.

``` r
preproc_test <- bake(prepped, test)
preds_test <- predict(fit_orig, new_data = preproc_test)
rmse_vec(preproc_test$Sale_Price, preds_test$.pred)
```

    ## [1] 0.1855153

Putting everything together
===========================

Using what we learned above, we can create a single function
`calculate_optimism_adjusted()` that performs all steps and returns the
adjusted model performance.

``` r
calculate_optimism_adjusted <- function(train_data, formula, preproc, n_resamples = 10L) {
  # Get apparent performance
  prepped <- prep(preproc, train_data)
  preproc_orig <- juice(prepped)
  fit_orig <- fit(linear_reg(), formula, preproc_orig)
  preds_orig <- predict(fit_orig, new_data = preproc_orig)
  perf_orig <- rmse_vec(last(preproc_orig), preds_orig$.pred)
  
  # Estimate optimism via bootstrap
  rsmpl <- bootstraps(train_data, times = n_resamples) %>% 
    mutate(
      processed = map(splits, ~ juice(prep(preproc, training(.)))),
      fitted = map(processed, ~ fit(linear_reg(), formula, data = .)),
      pred_app = map2(fitted, processed, ~ predict(.x, new_data = .y)),
      perf_app = map2_dbl(processed, pred_app, ~ rmse_vec(.x$Sale_Price, .y$.pred)),
      pred_test = map(fitted, ~ predict(., new_data = preproc_orig)),
      perf_test = map_dbl(pred_test, ~ rmse_vec(last(preproc_orig), .$.pred)),
      optim = perf_app - perf_test
    )
  
  mean_opt <- mean(rsmpl$optim)
  std_opt <- sd(rsmpl$optim)

  # Adjust for optimism
  tibble(
    .metric = "rmse",
    mean = perf_orig - mean_opt, 
    n = n_resamples, 
    std_err = std_opt / sqrt(n_resamples)
  )
}
```

We also define a similar function `eval_test` for the external
validation and wrappers around *tidymodel*’s `fit_resample` to do the
same for repeated cross-validation (`calculate_repeated_cv()`) and
standard bootstrap (`calculate_standard_bs()`), which we will compare in
a second.

``` r
eval_test <- function(train_data, test_data, formula, preproc) {
  
  prepped <- prep(preproc, train_data)
  preproc_train <- juice(prepped)
  preproc_test <- bake(prepped, test_data)
  fitted <- fit(linear_reg(), formula, data = preproc_train)
  preds <- predict(fitted, new_data = preproc_test)
  rmse_vec(preproc_test$Sale_Price, preds$.pred)
}

calculate_repeated_cv <- function(train_data, formula, preproc, v = 10L, repeats = 1L){
  rsmpl <- vfold_cv(train_data, v = v, repeats = repeats)

  show_best(fit_resamples(linear_reg(), preproc, rsmpl), metric = "rmse") %>% 
    select(-.estimator, -.config)
}

calculate_standard_bs <- function(train_data, formula, preproc, n_resamples = 10L) {
  rsmpl <- bootstraps(train_data, times = n_resamples, apparent = FALSE)

  show_best(fit_resamples(linear_reg(), preproc, rsmpl), metric = "rmse") %>% 
    select(-.estimator, -.config)
}
```

Comparison of validation methods
================================

In this last section, we will compare the results obtained from OAB to
two other well-known validation methods: repeated 10-fold
cross-validation and standard bootstrap. In the former, we randomly
split the data into 10 mutually exclusive folds of equal size. In a
round-robin fashion, we set aside one fold as an evaluation set and use
the remaining nine to train our model. We then choose the next fold and
do the same. After one round, we have ten estimates of model
performance, one for each held-out fold. We repeat this process several
times with new random seeds to get the same number of resamples as were
used for the bootstrap. With standard bootstrap, we fit our models on
the same bootstrapped data but evaluate them on the samples that were
randomly excluded from that particular bootstrap — similar to the
held-out fold of cross-validation.

In order to get a good comparison of methods, we won’t stick with a
single train-test split as before but use nested validation. The reason
for this is that our test data isn’t truly external. Instead, it is
randomly sampled from the entire development dataset (which in this case
was all of Ames). Holding out a single chunk of that data as test data
would be wasteful and could again result in us being particularly lucky
or unlucky in the selection of that chunk. This is particularly
problematic if we further perform hyperparameter searches. In nested
validation, we mitigate this risk by wrapping our entire internal
validation in another cross-validation loop, i.e., we treat the held-out
set of an outer cross-validation as the “external” test set.

``` r
outer <- vfold_cv(ames, v = 5, repeats = 1)

outer <- outer %>% 
  mutate(
    opt = splits %>% 
      map(~ calculate_optimism_adjusted(training(.), formula, preproc, 10L)),
    cv = splits %>% 
      map(~ calculate_repeated_cv(training(.), formula, preproc, repeats = 2L)), 
    bs = splits %>% 
      map(~ calculate_standard_bs(training(.), formula, preproc, 10L)), 
    test = splits %>% 
      map_dbl(~ eval_test(training(.), testing(.), formula, preproc))
  )
```

``` r
format_results <- function(outer, method) {
  method <- rlang::enquo(method)
  
  outer %>% 
    unnest(!!method) %>% 
    summarise(
      rsmpl_lower = mean(mean) - 2 * mean(std_err),
      rsmpl_mean  = mean(mean), 
      rsmpl_upper = mean(mean) + 2 * mean(std_err), 
      test_mean   = mean(test),
      p_within    = mean(rsmpl_lower < test_mean & test_mean < rsmpl_upper)
    )
}

tibble(method = c("opt", "cv", "bs")) %>% 
  bind_cols(bind_rows(
    format_results(outer, opt), 
    format_results(outer, cv), 
    format_results(outer, bs), 
  ))
```

    ## # A tibble: 3 x 6
    ##   method rsmpl_lower rsmpl_mean rsmpl_upper test_mean p_within
    ##   <chr>        <dbl>      <dbl>       <dbl>     <dbl>    <dbl>
    ## 1 opt          0.174      0.180       0.187     0.181        1
    ## 2 cv           0.160      0.177       0.194     0.181        1
    ## 3 bs           0.173      0.183       0.193     0.181        1

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-Harrell2015-ws" class="csl-entry">

Harrell, Frank E, Jr. 2015. *Regression Modeling Strategies: With
Applications to Linear Models, Logistic and Ordinal Regression, and
Survival Analysis*. Springer, Cham.

</div>

<div id="ref-Kuhn2022-al" class="csl-entry">

Kuhn, Max, and Julia Silge. 2022. *Tidy Modeling with r*.
<https://www.tmwr.org/>.

</div>

<div id="ref-Steyerberg2019-yc" class="csl-entry">

Steyerberg, Ewout W. 2019. *Clinical Prediction Models: A Practical
Approach to Development, Validation, and Updating*. Springer, Cham.

</div>

</div>
