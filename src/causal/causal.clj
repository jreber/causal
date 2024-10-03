(ns causal.causal
  (:require [causal.fcit :refer [dependent? independent?]]
            [scicloj.ml.dataset :as ds]
            [clojure.pprint :refer [pprint print-table]]))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; testing for various causal model shapes
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn- get-failed-conditions [ds conditions]
  (->> conditions
       (keep (fn [[pred-msg pred]]
               (when-not (pred ds)
                 pred-msg)))
       doall))

(defn collider-violations [ds {:keys [collider/child
                                      collider/parent1
                                      collider/parent2]}]
  (let [conditions {"parents should be independent"         #(independent? % parent1 parent2)
                    "parent1 and child should be dependent" #(dependent? % child parent1)
                    "parent2 and child should be dependent" #(dependent? % child parent2)
                    "parents should be dependent conditioned on child" #(dependent? % parent1 parent2 child)}]
    (get-failed-conditions ds conditions)))

(def collider? (comp empty? collider-violations))


(defn fork-violations [ds {:keys [fork/parent
                                  fork/child1
                                  fork/child2]}]
  (let [conditions {"children should be dependent"          #(dependent? % child1 child2)
                    "parent and child1 should be dependent" #(dependent? % child1 parent)
                    "parent and child2 should be dependent" #(dependent? % child2 parent)
                    "children should be independent conditioned on parent" #(independent? % child1 child2 parent)}]
    (get-failed-conditions ds conditions)))

(def fork? (comp empty? fork-violations))


(defn chain-violations [ds {:keys [chain/first
                                   chain/middle
                                   chain/last]}]
  (let [conditions {"first and middle should be dependent" #(dependent? % first middle)
                    "middle and last should be dependent"  #(dependent? % middle last)
                    "first and last should be dependent"   #(dependent? % first last)
                    "first and last should be independent conditioned on middle" #(independent? % first last middle)}]
    (get-failed-conditions ds conditions)))

(def chain? (comp empty? chain-violations))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; creating various causal model shapes
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn gen-independent []
  (map #(- % 0.5) (repeatedly rand)))

(defn make-fork-df
  ([]
   (make-fork-df 1000))
  ([num-rows]
   (let [Z (take num-rows (gen-independent))
         X (map + Z (gen-independent))
         Y (map + Z (gen-independent))]
     (ds/dataset
      {:x X
       :y Y
       :z Z}
      {:dataset-name "fork X<-Z->Y"}))))

(defn make-chain-df
  ([]
   (make-chain-df 1000))
  ([num-rows]
   (let [X (take num-rows (gen-independent))
         Y (map + X (gen-independent))
         Z (map + Y (gen-independent))]
     (ds/dataset
      {:x X
       :y Y
       :z Z}
      {:dataset-name "chain X->Y->X"}))))

(defn make-collider-df
  ([]
   (make-collider-df 1000))
  ([num-rows]
   (let [X (take num-rows (gen-independent))
         Y (take num-rows (gen-independent))
         Z (map + X Y)]
     (ds/dataset
      {:x X
       :y Y
       :z Z}
      {:dataset-name "collider X->Z<-Y"}))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; tests (todo move to real test file)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(let [collider-df (make-collider-df)
      chain-df (make-chain-df)
      fork-df (make-fork-df)]
  (assert (collider? collider-df {:collider/parent1 :x
                                  :collider/parent2 :y
                                  :collider/child :z}))
  (assert (chain? chain-df {:chain/first :x
                            :chain/middle :y
                            :chain/last :z}))
  (assert (fork? fork-df {:fork/child1 :x
                          :fork/child2 :y
                          :fork/parent :z})))
