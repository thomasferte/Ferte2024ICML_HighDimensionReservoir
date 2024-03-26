High dimension reservoir
================

<script src="results_high_dim_rc_files/libs/kePrint-0.0.1/kePrint.js"></script>
<link href="results_high_dim_rc_files/libs/lightable-0.0.1/lightable.css" rel="stylesheet" />


# Setting

![](results_high_dim_rc_files/figure-commonmark/unnamed-chunk-2-1.png)

The total number of features is 409.

# PCA

![](results_high_dim_rc_files/figure-commonmark/unnamed-chunk-4-1.png)

![](results_high_dim_rc_files/figure-commonmark/unnamed-chunk-5-1.png)

![](results_high_dim_rc_files/figure-commonmark/unnamed-chunk-6-1.png)

# Forecast

## Performance

<table class="table" style="margin-left: auto; margin-right: auto;">
<caption>Model performance</caption>
 <thead>
  <tr>
   <th style="text-align:left;"> model </th>
   <th style="text-align:left;"> update </th>
   <th style="text-align:right;"> AE </th>
   <th style="text-align:right;"> sd_AE </th>
   <th style="text-align:right;"> AE_baseline </th>
   <th style="text-align:right;"> sd_AE_baseline </th>
   <th style="text-align:right;"> RE </th>
   <th style="text-align:right;"> sd_RE </th>
   <th style="text-align:right;"> RE_baseline </th>
   <th style="text-align:right;"> sd_RE_baseline </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> Reservoir FS (GA - 2000) </td>
   <td style="text-align:left;"> No </td>
   <td style="text-align:right;"> 14.66 </td>
   <td style="text-align:right;"> 12.63 </td>
   <td style="text-align:right;"> -3.92 </td>
   <td style="text-align:right;"> 7.70 </td>
   <td style="text-align:right;"> 0.25 </td>
   <td style="text-align:right;"> 0.31 </td>
   <td style="text-align:right;"> 0.83 </td>
   <td style="text-align:right;"> 1.46 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Reservoir FS (GA) </td>
   <td style="text-align:left;"> No </td>
   <td style="text-align:right;"> 15.27 </td>
   <td style="text-align:right;"> 12.88 </td>
   <td style="text-align:right;"> -3.31 </td>
   <td style="text-align:right;"> 6.06 </td>
   <td style="text-align:right;"> 0.24 </td>
   <td style="text-align:right;"> 0.33 </td>
   <td style="text-align:right;"> 0.84 </td>
   <td style="text-align:right;"> 1.67 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Reservoir FS (GA) </td>
   <td style="text-align:left;"> Yes </td>
   <td style="text-align:right;"> 15.79 </td>
   <td style="text-align:right;"> 13.08 </td>
   <td style="text-align:right;"> -2.80 </td>
   <td style="text-align:right;"> 5.23 </td>
   <td style="text-align:right;"> 0.27 </td>
   <td style="text-align:right;"> 0.37 </td>
   <td style="text-align:right;"> 0.90 </td>
   <td style="text-align:right;"> 1.11 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Reservoir FS (GA - 10 esn) </td>
   <td style="text-align:left;"> No </td>
   <td style="text-align:right;"> 15.46 </td>
   <td style="text-align:right;"> 12.31 </td>
   <td style="text-align:right;"> -3.13 </td>
   <td style="text-align:right;"> 9.06 </td>
   <td style="text-align:right;"> 0.29 </td>
   <td style="text-align:right;"> 0.32 </td>
   <td style="text-align:right;"> 0.82 </td>
   <td style="text-align:right;"> 1.85 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Reservoir FS (GA - 10 esn) </td>
   <td style="text-align:left;"> Yes </td>
   <td style="text-align:right;"> 15.48 </td>
   <td style="text-align:right;"> 12.61 </td>
   <td style="text-align:right;"> -3.11 </td>
   <td style="text-align:right;"> 9.11 </td>
   <td style="text-align:right;"> 0.28 </td>
   <td style="text-align:right;"> 0.31 </td>
   <td style="text-align:right;"> 0.84 </td>
   <td style="text-align:right;"> 1.93 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Reservoir FS 7 days (GA) </td>
   <td style="text-align:left;"> No </td>
   <td style="text-align:right;"> 8.94 </td>
   <td style="text-align:right;"> 7.82 </td>
   <td style="text-align:right;"> -2.07 </td>
   <td style="text-align:right;"> 5.22 </td>
   <td style="text-align:right;"> 0.15 </td>
   <td style="text-align:right;"> 0.20 </td>
   <td style="text-align:right;"> 0.83 </td>
   <td style="text-align:right;"> 3.54 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Reservoir FS 21 days (GA) </td>
   <td style="text-align:left;"> No </td>
   <td style="text-align:right;"> 25.48 </td>
   <td style="text-align:right;"> 16.22 </td>
   <td style="text-align:right;"> 0.14 </td>
   <td style="text-align:right;"> 2.21 </td>
   <td style="text-align:right;"> 0.52 </td>
   <td style="text-align:right;"> 0.50 </td>
   <td style="text-align:right;"> 1.00 </td>
   <td style="text-align:right;"> 2.62 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Reservoir FS (GA) all anteriority </td>
   <td style="text-align:left;"> No </td>
   <td style="text-align:right;"> 15.71 </td>
   <td style="text-align:right;"> 12.69 </td>
   <td style="text-align:right;"> -2.88 </td>
   <td style="text-align:right;"> 6.90 </td>
   <td style="text-align:right;"> 0.28 </td>
   <td style="text-align:right;"> 0.33 </td>
   <td style="text-align:right;"> 0.86 </td>
   <td style="text-align:right;"> 1.72 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Reservoir FS (GA - PCA) </td>
   <td style="text-align:left;"> No </td>
   <td style="text-align:right;"> 15.59 </td>
   <td style="text-align:right;"> 12.90 </td>
   <td style="text-align:right;"> -3.00 </td>
   <td style="text-align:right;"> 7.70 </td>
   <td style="text-align:right;"> 0.28 </td>
   <td style="text-align:right;"> 0.33 </td>
   <td style="text-align:right;"> 0.87 </td>
   <td style="text-align:right;"> 1.76 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Reservoir FS (GA - PCA) </td>
   <td style="text-align:left;"> Yes </td>
   <td style="text-align:right;"> 15.63 </td>
   <td style="text-align:right;"> 12.67 </td>
   <td style="text-align:right;"> -2.95 </td>
   <td style="text-align:right;"> 6.37 </td>
   <td style="text-align:right;"> 0.29 </td>
   <td style="text-align:right;"> 0.35 </td>
   <td style="text-align:right;"> 0.85 </td>
   <td style="text-align:right;"> 1.68 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Reservoir FS (RS) </td>
   <td style="text-align:left;"> No </td>
   <td style="text-align:right;"> 18.45 </td>
   <td style="text-align:right;"> 13.27 </td>
   <td style="text-align:right;"> -0.14 </td>
   <td style="text-align:right;"> 1.13 </td>
   <td style="text-align:right;"> 0.36 </td>
   <td style="text-align:right;"> 0.38 </td>
   <td style="text-align:right;"> 1.00 </td>
   <td style="text-align:right;"> 0.31 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Reservoir FS (RS) </td>
   <td style="text-align:left;"> Yes </td>
   <td style="text-align:right;"> 17.81 </td>
   <td style="text-align:right;"> 13.34 </td>
   <td style="text-align:right;"> -0.77 </td>
   <td style="text-align:right;"> 2.50 </td>
   <td style="text-align:right;"> 0.31 </td>
   <td style="text-align:right;"> 0.38 </td>
   <td style="text-align:right;"> 0.98 </td>
   <td style="text-align:right;"> 0.62 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Reservoir no FS (GA) </td>
   <td style="text-align:left;"> No </td>
   <td style="text-align:right;"> 17.59 </td>
   <td style="text-align:right;"> 13.35 </td>
   <td style="text-align:right;"> -0.99 </td>
   <td style="text-align:right;"> 2.82 </td>
   <td style="text-align:right;"> 0.31 </td>
   <td style="text-align:right;"> 0.38 </td>
   <td style="text-align:right;"> 0.96 </td>
   <td style="text-align:right;"> 0.70 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Reservoir no FS (GA) </td>
   <td style="text-align:left;"> Yes </td>
   <td style="text-align:right;"> 16.95 </td>
   <td style="text-align:right;"> 13.50 </td>
   <td style="text-align:right;"> -1.64 </td>
   <td style="text-align:right;"> 4.50 </td>
   <td style="text-align:right;"> 0.29 </td>
   <td style="text-align:right;"> 0.38 </td>
   <td style="text-align:right;"> 0.97 </td>
   <td style="text-align:right;"> 1.53 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Elastic-net (RS) </td>
   <td style="text-align:left;"> No </td>
   <td style="text-align:right;"> 15.83 </td>
   <td style="text-align:right;"> 12.41 </td>
   <td style="text-align:right;"> -2.76 </td>
   <td style="text-align:right;"> 7.10 </td>
   <td style="text-align:right;"> 0.29 </td>
   <td style="text-align:right;"> 0.34 </td>
   <td style="text-align:right;"> 0.86 </td>
   <td style="text-align:right;"> 1.65 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Elastic-net (RS) </td>
   <td style="text-align:left;"> Yes </td>
   <td style="text-align:right;"> 16.22 </td>
   <td style="text-align:right;"> 12.54 </td>
   <td style="text-align:right;"> -2.36 </td>
   <td style="text-align:right;"> 8.50 </td>
   <td style="text-align:right;"> 0.30 </td>
   <td style="text-align:right;"> 0.38 </td>
   <td style="text-align:right;"> 0.88 </td>
   <td style="text-align:right;"> 1.29 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> XGB (RS) </td>
   <td style="text-align:left;"> No </td>
   <td style="text-align:right;"> 15.45 </td>
   <td style="text-align:right;"> 13.91 </td>
   <td style="text-align:right;"> -3.14 </td>
   <td style="text-align:right;"> 9.56 </td>
   <td style="text-align:right;"> 0.28 </td>
   <td style="text-align:right;"> 0.36 </td>
   <td style="text-align:right;"> 0.81 </td>
   <td style="text-align:right;"> 2.10 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> XGB (RS) </td>
   <td style="text-align:left;"> Yes </td>
   <td style="text-align:right;"> 16.32 </td>
   <td style="text-align:right;"> 14.01 </td>
   <td style="text-align:right;"> -2.27 </td>
   <td style="text-align:right;"> 9.89 </td>
   <td style="text-align:right;"> 0.29 </td>
   <td style="text-align:right;"> 0.46 </td>
   <td style="text-align:right;"> 0.89 </td>
   <td style="text-align:right;"> 2.40 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> XGB 21 days (RS) </td>
   <td style="text-align:left;"> No </td>
   <td style="text-align:right;"> 25.85 </td>
   <td style="text-align:right;"> 22.65 </td>
   <td style="text-align:right;"> 0.51 </td>
   <td style="text-align:right;"> 23.37 </td>
   <td style="text-align:right;"> 0.47 </td>
   <td style="text-align:right;"> 1.13 </td>
   <td style="text-align:right;"> 0.87 </td>
   <td style="text-align:right;"> NaN </td>
  </tr>
  <tr>
   <td style="text-align:left;"> XGB 7 days (RS) </td>
   <td style="text-align:left;"> No </td>
   <td style="text-align:right;"> 8.96 </td>
   <td style="text-align:right;"> 7.69 </td>
   <td style="text-align:right;"> -2.04 </td>
   <td style="text-align:right;"> 6.33 </td>
   <td style="text-align:right;"> 0.14 </td>
   <td style="text-align:right;"> 0.22 </td>
   <td style="text-align:right;"> 0.80 </td>
   <td style="text-align:right;"> 5.20 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> LSTM-PCA </td>
   <td style="text-align:left;"> No </td>
   <td style="text-align:right;"> 15.74 </td>
   <td style="text-align:right;"> 13.98 </td>
   <td style="text-align:right;"> -2.84 </td>
   <td style="text-align:right;"> 11.80 </td>
   <td style="text-align:right;"> 0.24 </td>
   <td style="text-align:right;"> 0.50 </td>
   <td style="text-align:right;"> 0.76 </td>
   <td style="text-align:right;"> 3.32 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> LSTM-PCA </td>
   <td style="text-align:left;"> Yes </td>
   <td style="text-align:right;"> 15.74 </td>
   <td style="text-align:right;"> 13.98 </td>
   <td style="text-align:right;"> -2.84 </td>
   <td style="text-align:right;"> 11.80 </td>
   <td style="text-align:right;"> 0.24 </td>
   <td style="text-align:right;"> 0.50 </td>
   <td style="text-align:right;"> 0.76 </td>
   <td style="text-align:right;"> 3.32 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> LSTM-RC-GA </td>
   <td style="text-align:left;"> No </td>
   <td style="text-align:right;"> 15.08 </td>
   <td style="text-align:right;"> 12.71 </td>
   <td style="text-align:right;"> -3.51 </td>
   <td style="text-align:right;"> 9.30 </td>
   <td style="text-align:right;"> 0.23 </td>
   <td style="text-align:right;"> 0.48 </td>
   <td style="text-align:right;"> 0.80 </td>
   <td style="text-align:right;"> 1.76 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> LSTM-RC-GA </td>
   <td style="text-align:left;"> Yes </td>
   <td style="text-align:right;"> 15.08 </td>
   <td style="text-align:right;"> 12.71 </td>
   <td style="text-align:right;"> -3.51 </td>
   <td style="text-align:right;"> 9.30 </td>
   <td style="text-align:right;"> 0.23 </td>
   <td style="text-align:right;"> 0.48 </td>
   <td style="text-align:right;"> 0.80 </td>
   <td style="text-align:right;"> 1.76 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Transformers-PCA </td>
   <td style="text-align:left;"> No </td>
   <td style="text-align:right;"> 19.28 </td>
   <td style="text-align:right;"> 16.80 </td>
   <td style="text-align:right;"> 0.69 </td>
   <td style="text-align:right;"> 11.64 </td>
   <td style="text-align:right;"> 0.32 </td>
   <td style="text-align:right;"> 0.51 </td>
   <td style="text-align:right;"> 1.00 </td>
   <td style="text-align:right;"> 2.79 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Transformers-PCA </td>
   <td style="text-align:left;"> Yes </td>
   <td style="text-align:right;"> 19.28 </td>
   <td style="text-align:right;"> 16.80 </td>
   <td style="text-align:right;"> 0.69 </td>
   <td style="text-align:right;"> 11.64 </td>
   <td style="text-align:right;"> 0.32 </td>
   <td style="text-align:right;"> 0.51 </td>
   <td style="text-align:right;"> 1.00 </td>
   <td style="text-align:right;"> 2.79 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Transformers-RC-GA </td>
   <td style="text-align:left;"> No </td>
   <td style="text-align:right;"> 18.12 </td>
   <td style="text-align:right;"> 15.43 </td>
   <td style="text-align:right;"> -0.47 </td>
   <td style="text-align:right;"> 9.33 </td>
   <td style="text-align:right;"> 0.31 </td>
   <td style="text-align:right;"> 0.44 </td>
   <td style="text-align:right;"> 0.97 </td>
   <td style="text-align:right;"> 4.02 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Transformers-RC-GA </td>
   <td style="text-align:left;"> Yes </td>
   <td style="text-align:right;"> 18.12 </td>
   <td style="text-align:right;"> 15.43 </td>
   <td style="text-align:right;"> -0.47 </td>
   <td style="text-align:right;"> 9.33 </td>
   <td style="text-align:right;"> 0.31 </td>
   <td style="text-align:right;"> 0.44 </td>
   <td style="text-align:right;"> 0.97 </td>
   <td style="text-align:right;"> 4.02 </td>
  </tr>
</tbody>
</table>

## RC aggregation

![Aggregation of
reservoir](results_high_dim_rc_files/figure-commonmark/rc-aggreg-1.png)

## Graphical forecast

![Model
prediction](results_high_dim_rc_files/figure-commonmark/model-prediction-1.png)

![All model prediction depending on
update](results_high_dim_rc_files/figure-commonmark/model-prediction-all-1.png)

# Hyperparameters

## Numeric hyperparameters

![Numeric
hyperparameters](results_high_dim_rc_files/figure-commonmark/numeric-hp-1.png)

![Numeric hyperparameters of
RC-GA](results_high_dim_rc_files/figure-commonmark/numeric-hp-rc-ga-1.png)

## Categorical hyperparameters

    [[1]]
    NULL

    [[2]]
    NULL

    [[1]]
    NULL

    [[2]]
    NULL

![Feature selection by
RC-GA](results_high_dim_rc_files/figure-commonmark/feature-selection-rc-ga-1.png)

![Feature
selection](results_high_dim_rc_files/figure-commonmark/feature-selection-all-1.png)

![Feature
selection](results_high_dim_rc_files/figure-commonmark/feature-selection-all-2.png)

## Ridge update of 2000 units reservoir

![Ridge hyperparameter of 2000 units
RC-GA](results_high_dim_rc_files/figure-commonmark/ridge-for-2000-units-1.png)

# Failure of monthly update

<table class="table" style="margin-left: auto; margin-right: auto;">
<caption>Performance by leaking rate</caption>
 <thead>
  <tr>
   <th style="text-align:left;"> model </th>
   <th style="text-align:left;"> update </th>
   <th style="text-align:left;"> leaking_rate </th>
   <th style="text-align:right;"> n </th>
   <th style="text-align:right;"> AE </th>
   <th style="text-align:right;"> AE_baseline </th>
   <th style="text-align:right;"> RE </th>
   <th style="text-align:right;"> RE_baseline </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> Reservoir FS (GA) </td>
   <td style="text-align:left;"> No </td>
   <td style="text-align:left;"> &lt; 1e-3 </td>
   <td style="text-align:right;"> 14 </td>
   <td style="text-align:right;"> 15.07126 </td>
   <td style="text-align:right;"> -3.515306 </td>
   <td style="text-align:right;"> 0.2580806 </td>
   <td style="text-align:right;"> 0.8247984 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Reservoir FS (GA) </td>
   <td style="text-align:left;"> No </td>
   <td style="text-align:left;"> &gt; 1e-3 </td>
   <td style="text-align:right;"> 26 </td>
   <td style="text-align:right;"> 15.51757 </td>
   <td style="text-align:right;"> -3.068987 </td>
   <td style="text-align:right;"> 0.2560384 </td>
   <td style="text-align:right;"> 0.8473388 </td>
  </tr>
</tbody>
</table>

![Prediction by leaking
rate](results_high_dim_rc_files/figure-commonmark/prediction-by-lr-1.png)

# Computing time

<!-- ```{r} -->
<!-- #| eval: false -->
<!-- ### redo on local machine -->
<!-- df_time_hp <- df_all_hp_best40 %>% -->
<!--   slice_min(date) %>% -->
<!--   filter(model %in% c("Reservoir FS (RS)", -->
<!--                       "Elastic-net (RS)", -->
<!--                       "XGB (RS)")) %>% -->
<!--   mutate(nb_features = select(., ends_with("_bin")) %>% mutate_all(.funs = function(x) x == "y") %>% rowSums()) %>% -->
<!--   select(model, time_seconds, all_of(numeric_hp), nb_features) %>% -->
<!--   tidyr::pivot_longer(cols = -c("model", "time_seconds")) %>% -->
<!--   na.omit() -->
<!-- read.csv("data_obfuscated/20210329.csv") %>% -->
<!--   select(-outcome, -outcomeDate, -START_DATE) %>% -->
<!--   dim() -->
<!-- plot_time_hp <- df_time_hp %>% -->
<!--   ggplot(mapping = aes(x = value, y = time_seconds)) + -->
<!--   geom_point(size = 0.01) + -->
<!--   scale_x_log10() + -->
<!--   facet_wrap(model ~ name, scales = "free", ncol = 3) + -->
<!--   theme_minimal() -->
<!-- df_time_hp %>% -->
<!--   filter(name == "nb_features") %>% -->
<!--   reframe(quantile(value, probs = c(0.25, 0.5, 0.75))) -->
<!-- ggsave(here::here("results/final_figures/plot_time_hp.pdf"), -->
<!--        plot = plot_time_hp, -->
<!--        useDingbats = TRUE, -->
<!--        height = 7, -->
<!--        width = 7) -->
<!-- df_all_hp %>% -->
<!--   slice_min(date) %>% -->
<!--   group_by(model) %>% -->
<!--   summarise(q1 = quantile(time_seconds, 0.25), -->
<!--             median = quantile(time_seconds, 0.5), -->
<!--             q3 = quantile(time_seconds, 0.75)) -->
<!-- ``` -->

We evaluated the time needed to train and forecast on January 2022
month.

<table class="table" style="margin-left: auto; margin-right: auto;">
<caption>Time on january month by algorithm. For RCGA, 40 RC where performed.</caption>
 <thead>
  <tr>
   <th style="text-align:left;"> Scenario </th>
   <th style="text-align:right;"> Time (s) </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> Elastic-net (RS) </td>
   <td style="text-align:right;"> 0.2458057 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> XGB (RS) </td>
   <td style="text-align:right;"> 74.4722674 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Reservoir FS (GA) </td>
   <td style="text-align:right;"> 145.5468154 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Reservoir FS (GA - 2000) </td>
   <td style="text-align:right;"> 290.6992917 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> LSTM </td>
   <td style="text-align:right;"> 32.2258518 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Transformers </td>
   <td style="text-align:right;"> 868.7015507 </td>
  </tr>
</tbody>
</table>

# Reservoir vs raw features importance

# Number of parameters

<table class="table" style="margin-left: auto; margin-right: auto;">
<caption>Number of parameters</caption>
 <thead>
  <tr>
   <th style="text-align:left;"> model </th>
   <th style="text-align:right;"> min(nb_param) </th>
   <th style="text-align:right;"> max(nb_param) </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> RCGA </td>
   <td style="text-align:right;"> 682 </td>
   <td style="text-align:right;"> 725 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> RCGAPCA </td>
   <td style="text-align:right;"> 508 </td>
   <td style="text-align:right;"> 711 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Enet </td>
   <td style="text-align:right;"> 394 </td>
   <td style="text-align:right;"> 394 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> XGB </td>
   <td style="text-align:right;"> 12842 </td>
   <td style="text-align:right;"> 12842 </td>
  </tr>
</tbody>
</table>

<table class="table" style="margin-left: auto; margin-right: auto;">
<caption>Number of features</caption>
 <thead>
  <tr>
   <th style="text-align:left;"> model </th>
   <th style="text-align:right;"> q25 </th>
   <th style="text-align:right;"> q50 </th>
   <th style="text-align:right;"> q75 </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> RCGA </td>
   <td style="text-align:right;"> 191 </td>
   <td style="text-align:right;"> 197.0 </td>
   <td style="text-align:right;"> 204 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> RCGAPCA </td>
   <td style="text-align:right;"> 12 </td>
   <td style="text-align:right;"> 14.0 </td>
   <td style="text-align:right;"> 18 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Enet </td>
   <td style="text-align:right;"> 228 </td>
   <td style="text-align:right;"> 233.0 </td>
   <td style="text-align:right;"> 236 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> XGB </td>
   <td style="text-align:right;"> 392 </td>
   <td style="text-align:right;"> 393.5 </td>
   <td style="text-align:right;"> 396 </td>
  </tr>
</tbody>
</table>

# Performance by date update

## Confounding matrix

![Performance by date
update](results_high_dim_rc_files/figure-commonmark/performance-by-date-update-1.png)

## Towards more agile update

<table class="table" style="margin-left: auto; margin-right: auto;">
<caption>Performance while selecting best hp sets among all of the best hp sets of very month. Results depending on the evaluation period</caption>
 <thead>
  <tr>
   <th style="text-align:right;"> Anteriority </th>
   <th style="text-align:right;"> MAE </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:right;"> 0 </td>
   <td style="text-align:right;"> 15.47284 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 1 </td>
   <td style="text-align:right;"> 15.47284 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 3 </td>
   <td style="text-align:right;"> 15.46620 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 7 </td>
   <td style="text-align:right;"> 15.61123 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 14 </td>
   <td style="text-align:right;"> 15.76887 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> 28 </td>
   <td style="text-align:right;"> 15.74196 </td>
  </tr>
  <tr>
   <td style="text-align:right;"> Inf </td>
   <td style="text-align:right;"> 15.88550 </td>
  </tr>
</tbody>
</table>

![Forecast based on agile hp update where best hyperparameters from
every 40 best reservoir depending on the evaluation
period](results_high_dim_rc_files/figure-commonmark/performance-agile-update-1.png)
