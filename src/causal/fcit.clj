(ns causal.fcit
  "Implementation of the Fast Conditional Independence Test (FCIT).

  Tests X ⊥ Y | Z (or X ⊥ Y | Z1, Z2, …) by comparing two gradient-boosted
  models on the same train/test split:

    D1  — trained on (X, Z…): the real model.
    D0  — trained on (shuffled-X, Z…) [marginal] or (Z… only) [conditional]:
          the null model, where X carries no information about Y beyond Z….

  For each random split we record the ratio D0/D1.  When X is informative
  the ratio is systematically > 1.  A one-sample t-test (H1: mean > 1) on
  the ratio samples yields the p-value.  This matches Chalupka et al. (2018)."
  (:require [scicloj.ml.dataset :as ds]
            [scicloj.ml.core :as ml]
            [scicloj.ml.metamorph :as mm]
            [fastmath.stats :as stats]))

;; ---------------------------------------------------------------------------
;; Internal helpers
;; ---------------------------------------------------------------------------

(defn- create-pipeline
  "Gradient-boosted regression pipeline predicting (first cols) from the rest."
  [& cols]
  (ml/pipeline
   (mm/select-columns cols)
   (mm/set-inference-target (first cols))
   (mm/model {:model-type :smile.regression/gradient-tree-boost
              :trees 100})))

(defn- compute-mse
  "Fits pipe-fn on train-ds, evaluates on test-ds, returns MSE."
  [pipe-fn train-ds test-ds]
  (let [trained-ctx (pipe-fn {:metamorph/data train-ds
                              :metamorph/mode :fit})
        test-ctx    (pipe-fn (assoc trained-ctx
                                    :metamorph/data test-ds
                                    :metamorph/mode :transform))
        target      (-> test-ctx :metamorph/data ds/column-names first)]
    (ml/mse (ds/->array (:metamorph/data test-ctx) target)
            (ds/->array test-ds target))))

(defn- ratio-sample
  "Returns D0/D1 on one shared random train/test split."
  [ds null-pipe real-pipe]
  (let [{:keys [train-ds test-ds]} (ds/train-test-split ds)
        d1 (compute-mse real-pipe train-ds test-ds)
        d0 (compute-mse null-pipe train-ds test-ds)]
    (let [ratio (/ d0 d1)]
      (cond
        (Double/isNaN ratio)      1.0
        (Double/isInfinite ratio) 1e6
        :else                     ratio))))

(defn- ratio-p-value
  "One-sample t-test on ratio samples: H1 = mean(ratios) > 1."
  [ratios]
  (:p-value (stats/t-test-one-sample ratios {:mu 1.0 :sides :one-sided-greater})))

;; ---------------------------------------------------------------------------
;; Public API
;; ---------------------------------------------------------------------------

(defn dependent?
  "Returns true if `target` and `predictor` are dependent (optionally
  conditioned on one or more `others`); false otherwise.

  Marginal case (no `others`): constructs the null by shuffling the predictor
  column on each trial, breaking its relationship with the target.

  Conditional case (`others` given): the null model predicts `target` from
  `others` alone; the real model adds `predictor`.  Both use the same
  train/test split per trial so that per-split noise cancels in the ratio.
  Accepts any number of conditioning variables, enabling d-separation tests
  that require blocking multiple paths simultaneously.

  Uses 10 trials and α = 0.05."
  ([ds target predictor]
   (let [cleaned   (ds/drop-missing ds [target predictor])
         null-col  :__fcit_null__
         real-pipe (create-pipeline target predictor)
         null-pipe (create-pipeline target null-col)
         n-trials  10
         ratios    (->> (range n-trials)
                        (pmap (fn [_]
                                (let [shuffled  (shuffle (vec (ds/->array cleaned predictor)))
                                      trial-ds  (ds/add-column cleaned null-col shuffled)
                                      {train :train-ds test :test-ds} (ds/train-test-split trial-ds)]
                                  (/ (compute-mse null-pipe train test)
                                     (compute-mse real-pipe train test)))))
                        doall)]
     (<= (ratio-p-value ratios) 0.05)))
  ([ds target predictor & others]
   (let [cols      (into [target predictor] others)
         cleaned   (ds/drop-missing ds cols)
         real-pipe (apply create-pipeline cols)
         null-pipe (apply create-pipeline (into [target] others))
         n-trials  10
         ratios    (->> (range n-trials)
                        (pmap (fn [_] (ratio-sample cleaned null-pipe real-pipe)))
                        doall)]
     (<= (ratio-p-value ratios) 0.05))))

(def independent?
  "Returns true if `target` and `predictor` are independent (optionally
  conditioned on one or more `others`); false otherwise."
  (complement dependent?))

(defn mse-samples
  "Returns `num-samples` D0/D1 ratio samples for the conditional model
  (target ~ predictor + others vs. target ~ others), useful for debugging
  signal strength before running a full test.

  `others` is a seq of one or more conditioning column names."
  [ds target predictor num-samples & others]
  (let [cols      (into [target predictor] others)
        cleaned   (ds/drop-missing ds cols)
        real-pipe (apply create-pipeline cols)
        null-pipe (apply create-pipeline (into [target] others))]
    (->> (range num-samples)
         (pmap (fn [_] (ratio-sample cleaned null-pipe real-pipe)))
         doall)))
