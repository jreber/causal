(ns causal.fcit
  (:require [clojure.pprint :refer [pprint print-table]]
            [scicloj.ml.dataset :as ds]
            [tech.v3.datatype.functional :as dfn]
            [scicloj.ml.core :as ml]
            [scicloj.ml.metamorph :as mm]
            [scicloj.ml.metamorph :as prep]
            [fastmath.stats :as stats]))


(def ds-name ds/dataset-name)

(defn create-pipeline [& cols]
  (ml/pipeline
   (mm/select-columns cols)
   (mm/set-inference-target (first cols))
   (mm/model {:model-type :smile.regression/gradient-tree-boost
              :trees 50})))

(defn compute-mse [pipe-fn train-ds test-ds]
  (let [trained-ctx (pipe-fn {:metamorph/data train-ds
                              :metamorph/mode :fit})
        test-ctx (pipe-fn
                  (assoc trained-ctx
                         :metamorph/data test-ds
                         :metamorph/mode :transform))
        target (-> test-ctx
                   :metamorph/data
                   ds/column-names
                   first)]
    (ml/mse (ds/->array (:metamorph/data test-ctx) target)
            (ds/->array test-ds target))))

(defn- compute-mses [ds [cols1 pipe-fn1] [cols2 pipe-fn2]]
  (let [{:keys [train-ds test-ds]} (ds/train-test-split ds)]
    {cols1 (compute-mse pipe-fn1 train-ds test-ds)
     cols2 (compute-mse pipe-fn2 train-ds test-ds)}))

(defn- t-test [cols1 cols2 ms]
  (println "Performing t-test on values:")
  (print-table ms)
  (stats/t-test-two-samples
   (map #(get % cols1) ms)
   (map #(get % cols2) ms)
   {:sides #_:both :one-sided-greater}))

(defn equally-good-predictors? [ds target-and-predictors1 target-and-predictors2]
  (let [cleaned-ds (ds/drop-missing ds (dedupe
                                        (concat target-and-predictors1 target-and-predictors2)))
        num-trials 20
        pipe-fn1 (apply create-pipeline target-and-predictors1)
        pipe-fn2 (apply create-pipeline target-and-predictors2)]
    (->> (range num-trials)
         (pmap (fn [_] (compute-mses cleaned-ds
                                     [target-and-predictors1 pipe-fn1]
                                     [target-and-predictors2 pipe-fn2])))
         (t-test target-and-predictors1 target-and-predictors2)
         :p-value)))

(defn dependent?
  ([ds target predictor]
   (-> ds
       (ds/add-column :ignore 0 :cycle)
       (dependent? target predictor :ignore)))
  ([ds target predictor other]
   (let [p-value (equally-good-predictors? ds
                                           [target other]
                                           [target predictor other])]
     (<= p-value 0.05))))

(def independent?
  (complement dependent?))
