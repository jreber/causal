(ns causal.causal
  (:require [causal.fcit :refer [dependent? independent?]]
            [scicloj.ml.dataset :as ds]))

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
                                      collider/parent2
                                      collider/decouplers]}]
  "Tests the collider structure parent1 → child ← parent2.

  Standard case (no :collider/decouplers): parents must be marginally
  independent, and become dependent when conditioning on the child.

  Generalised case (:collider/decouplers given): a d-separating set S that
  renders the parents conditionally independent.  The test then checks:
    parent1 ⊥ parent2 | S            (decouplers block all other paths)
    parent1 ⊥̸ parent2 | S ∪ {child}  (child opens the collider path)
  This is Pearl's d-separation applied to a conditioning set S ≠ ∅,
  equivalent to the standard test when S = ∅."
  (let [decouplers-vec (vec (or decouplers []))
        conditions
        {"parent1 and child should be dependent"
         #(dependent? % child parent1)
         "parent2 and child should be dependent"
         #(dependent? % child parent2)
         "parents should be independent given decouplers"
         #(apply independent? % parent1 parent2 decouplers-vec)
         "parents should be dependent given decouplers and child"
         #(apply dependent? % parent1 parent2 (conj decouplers-vec child))}]
    (get-failed-conditions ds conditions)))

(def collider? (comp empty? collider-violations))

(defn fork-violations [ds {:keys [fork/parent fork/parents fork/child1 fork/child2]}]
  "Tests the fork structure child1 ← parent(s) → child2.

  Accepts either :fork/parent (single keyword, backward-compatible) or
  :fork/parents (keyword or vector of keywords).  With multiple parents the
  test reflects two distinct overlapping forks sharing the same children;
  the key d-separation claim is child1 ⊥ child2 | {all parents}."
  (let [parents-vec (cond
                      (some? parents) (if (sequential? parents) (vec parents) [parents])
                      (some? parent)  [parent]
                      :else (throw (ex-info "fork spec requires :fork/parent or :fork/parents" {})))
        parent-dep-conditions
        (into {} (for [p      parents-vec
                       [lbl c] [["child1" child1] ["child2" child2]]]
                   [(str p " and " lbl " should be dependent")
                    (fn [d] (dependent? d c p))]))
        conditions
        (merge {"children should be dependent"
                #(dependent? % child1 child2)
                "children should be independent conditioned on all parents"
                #(apply independent? % child1 child2 parents-vec)}
               parent-dep-conditions)]
    (get-failed-conditions ds conditions)))

(def fork? (comp empty? fork-violations))

(defn chain-violations [ds {:keys [chain/first
                                   chain/middle
                                   chain/last
                                   chain/confounders]}]
  "Tests the chain structure first → middle → last.

  Accepts an optional :chain/confounders (keyword or vector of keywords) for
  variables that create additional active paths between first and last.
  The key d-separation claim becomes first ⊥ last | {middle, confounders},
  blocking all paths — not just the direct chain — between the endpoints."
  (let [confounders-vec (vec (or (when confounders
                                   (if (sequential? confounders)
                                     confounders
                                     [confounders]))
                                 []))
        blocking-set (into [middle] confounders-vec)
        conditions
        {"first and middle should be dependent"
         #(dependent? % first middle)
         "middle and last should be dependent"
         #(dependent? % middle last)
         "first and last should be dependent"
         #(dependent? % first last)
         "first and last should be independent conditioned on middle (and confounders)"
         #(apply independent? % first last blocking-set)}]
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
      {:dataset-name "chain X->Y->Z"}))))

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

