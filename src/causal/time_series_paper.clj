(ns causal.time-series-paper
  (:require [causal.fcit :as fcit]
            [scicloj.ml.dataset :as ds]
            [tablecloth.api :as tc]
            [tablecloth.column.api :as tcc]
            [tech.v3.dataset.column :as dsc]
            [clojure.pprint :refer [pprint print-table]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; the utils
;; see https://scicloj.github.io/tablecloth/#access-manipulation
;; and https://cljdoc.org/d/scicloj/tablecloth/7.029.2/api/tablecloth.column.api#shift
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defprotocol TimeSeries
  (shift [data w]))

(extend-type tech.v3.dataset.impl.column.Column
  TimeSeries
  (shift [col w]
    (cond
      (pos? w) (-> col
                   (dsc/extend-column-with-empty w)
                   (tcc/slice w :end))
      (neg? w) (-> col
                   (dsc/prepend-column-with-empty (- w))
                   (tcc/slice :start (dec (count col))))
      :default col)))

(extend-type tech.v3.dataset.impl.dataset.Dataset
  TimeSeries
  (shift [ds w]
    (->> (tc/columns ds :as-map)
         (reduce-kv
          (fn [m col-name col]
            (assoc m col-name (shift col w)))
          {})
         tc/dataset)))

(defn min-lag
  ([ds y x]
   (min-lag (get ds y)
            (get ds x)))
  ([y x]
   (loop [w 1]
     (let [shifted-x (shift x w)
           temp-ds (tc/dataset {:y y
                                :x shifted-x})]
       (cond
         (fcit/dependent? temp-ds :y :x) w
         (< 10 w) 0 #_(throw (ex-info "Could not find dependence within ten time steps"
                                      {:x x
                                       :y y}))
         :default (recur (inc w)))))))

(defn ->ds [& cols-or-ds]
  (->> cols-or-ds
       (mapcat #(if (tc/dataset? %)
                  (tc/columns %)
                  [%]))
       tc/dataset))

(defn ->dependent-args [target predictor & cols-or-ds]
  (let [conditioning-set (apply ->ds cols-or-ds)
        ds (->ds (tc/dataset {:y target
                              :x predictor})
                 conditioning-set)]
    (concat [ds :y :x]
            (keys conditioning-set))))

(defn dependent? [& args]
  (println "Testing for independence:" args)
  (apply fcit/dependent?
         (apply ->dependent-args args)))

(def independent? (complement dependent?))

(defn conditioning-set [Xs w X_i]
  (tc/dataset
   (for [X_j (tc/columns
              (tc/drop-columns Xs (dsc/column-name X_i)))]
     (shift X_j
            (- (w X_i)
               (w X_j)
               1)))))

(defn find-sg-unconfounded [Y Xs]
  (let [w (memoize (partial min-lag Y))
        shifted #(shift %1 (- %2))
        causes? (fn [X_i]
                  (let [S_i (conditioning-set Xs w X_i)
                        wi  (w X_i)]
                    (and (pos? wi)
                         (dependent?   (shifted Y wi) X_i              S_i (shifted Y (- wi 1)))
                         (independent? (shifted Y wi) (shifted X_i -1) S_i (shifted Y (- wi 1)) X_i))))]
    (->> Xs
         tc/columns
         (filter causes?)
         (map dsc/column-name)
         set)))

(defn find-sg-unconfounded-v2 [ds target-name]
  (find-sg-unconfounded
   (get ds target-name)
   (tc/drop-columns ds target-name)))
