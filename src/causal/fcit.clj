(ns causal.fcit
  (:require [clojure.pprint :refer [pprint print-table]]
            [scicloj.ml.dataset :as ds]
            [tech.v3.datatype.functional :as dfn]
            [scicloj.ml.core :as ml]
            [scicloj.ml.metamorph :as mm]
            [fastmath.stats :as stats]))


(def ds-name ds/dataset-name)

(defn compute-mse [pipe-fn train-ds test-ds target]
  (let [trained-ctx (pipe-fn {:metamorph/data train-ds
                              :metamorph/mode :fit})
        test-ctx (pipe-fn
                  (assoc trained-ctx
                         :metamorph/data test-ds
                         :metamorph/mode :transform))]
    (ml/mse (ds/->array (:metamorph/data test-ctx) target)
            (ds/->array test-ds target))))

(defn- compute-mses [ds pipe-fn-without-predictor pipe-fn-with-predictor target]
  (let [{:keys [train-ds test-ds]} (ds/train-test-split ds)]
    {:mse/with-predictor (compute-mse pipe-fn-with-predictor
                                      train-ds
                                      test-ds
                                      target)
     :mse/without-predictor (compute-mse pipe-fn-without-predictor
                                         train-ds
                                         test-ds
                                         target)}))

(defn- t-test
  ([ms]
   (t-test (map :mse/without-predictor ms)
           (map :mse/with-predictor ms)))
  ([without-predictor with-predictor]
   (stats/t-test-two-samples with-predictor without-predictor {:sides :one-sided-lower})))

(defn independent-prob
  ([ds target predictor]
   (let [target-avg (dfn/mean (ds/column ds target))]
     (-> ds
         (ds/add-column :ignore target-avg :cycle)
         (independent-prob target predictor :ignore))))
  ([ds target predictor other]
   (let [num-trials 8
         pipe-fn-with-predictor (ml/pipeline
                                 (mm/select-columns [target predictor other])
                                 (mm/set-inference-target target)
                                 (mm/model {:model-type :smile.regression/gradient-tree-boost}))
         pipe-fn-without-predictor (ml/pipeline
                                    (mm/select-columns [target other])
                                    (mm/set-inference-target target)
                                    (mm/model {:model-type :smile.regression/gradient-tree-boost}))
         t-test-result (->> (range num-trials)
                            (pmap (fn [_] (compute-mses ds pipe-fn-without-predictor pipe-fn-with-predictor target)))
                            t-test)]
     (->> t-test-result
          :p-value))))

(defn dependent? [& args]
  (<= (apply independent-prob args) 0.05))

(def independent?
  (complement dependent?))
