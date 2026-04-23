(ns causal.fcit
  "Implementation of the Fast Conditional Independence Test (FCIT).

  Tests X ⊥ Y | Z by comparing two gradient-boosted models on the same
  train/test split:

    D1  — trained on (X, Z): the real model.
    D0  — trained on (shuffled-X, Z) [marginal] or (Z only) [conditional]:
          the null model, where X carries no information about Y beyond Z.

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
    (/ d0 d1)))

(defn- ratio-p-value
  "One-sample t-test on ratio samples: H1 = mean(ratios) > 1."
  [ratios]
  (:p-value (stats/t-test-one-sample ratios {:mu 1.0 :sides :one-sided-greater})))

;; ---------------------------------------------------------------------------
;; Public API
;; ---------------------------------------------------------------------------

(defn dependent?
  "Returns true if `target` and `predictor` are dependent (optionally
  conditioned on `other`); false otherwise.

  Marginal case (no `other`): constructs the null by shuffling the predictor
  column on each trial, breaking its relationship with the target.

  Conditional case (`other` given): the null model predicts `target` from
  `other` alone; the real model adds `predictor`.  Both use the same
  train/test split per trial so that per-split noise cancels in the ratio.

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
  ([ds target predictor other]
   (let [cleaned   (ds/drop-missing ds [target predictor other])
         real-pipe (create-pipeline target predictor other)
         null-pipe (create-pipeline target other)
         n-trials  10
         ratios    (->> (range n-trials)
                        (pmap (fn [_] (ratio-sample cleaned null-pipe real-pipe)))
                        doall)]
     (<= (ratio-p-value ratios) 0.05))))

(def independent?
  "Returns true if `target` and `predictor` are independent (optionally
  conditioned on `other`); false otherwise."
  (complement dependent?))

(defn mse-samples
  "Returns `num-samples` D0/D1 ratio samples for the given model, useful
  for debugging the signal strength before running a full test."
  [ds target predictor other num-samples]
  (let [cleaned   (ds/drop-missing ds [target predictor other])
        real-pipe (create-pipeline target predictor other)
        null-pipe (create-pipeline target other)]
    (->> (range num-samples)
         (pmap (fn [_] (ratio-sample cleaned null-pipe real-pipe)))
         doall)))
