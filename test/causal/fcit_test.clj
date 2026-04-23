(ns causal.fcit-test
  "Tests for the Fast Conditional Independence Test (FCIT) implementation.

  These are statistical tests that use ML model comparison, so they may
  very rarely fail due to random variation. Use larger datasets and more
  trials for robustness."
  (:require [clojure.test :refer :all]
            [scicloj.ml.dataset :as ds]
            [causal.fcit :refer [dependent? independent?]]))

;; ---------------------------------------------------------------------------
;; Test data generators
;; ---------------------------------------------------------------------------

(defn- linear-dependent-data
  "Generates n rows where y = 2*x + small-noise. Strong linear dependence."
  [n]
  (let [x (repeatedly n rand)
        ;; Signal-to-noise ratio 4:1 so gradient boost reliably detects it
        y (map #(+ (* 4 %) (* 0.5 (rand))) x)]
    (ds/dataset {:x x :y y})))

(defn- independent-data
  "Generates n rows where x and y are completely independent draws."
  [n]
  (ds/dataset {:x (repeatedly n rand)
               :y (repeatedly n rand)}))

(defn- fork-data
  "Fork Z -> X, Z -> Y. X and Y are marginally dependent but conditionally
  independent given Z."
  [n]
  (let [z (repeatedly n rand)
        x (map #(+ % (rand)) z)
        y (map #(+ % (rand)) z)]
    (ds/dataset {:x x :y y :z z})))

(defn- collider-data
  "Collider X -> Z <- Y. X and Y are marginally independent but conditionally
  dependent given Z."
  [n]
  (let [x (repeatedly n rand)
        y (repeatedly n rand)
        z (map + x y)]
    (ds/dataset {:x x :y y :z z})))

;; ---------------------------------------------------------------------------
;; Tests: marginal dependence / independence
;; ---------------------------------------------------------------------------

(deftest dependent?-detects-strong-linear-relationship
  (testing "strongly correlated variables are flagged as dependent"
    (let [df (linear-dependent-data 1000)]
      (is (dependent? df :y :x)
          "y = 4x + noise, so x should explain y well"))))

(deftest independent?-detects-unrelated-variables
  (testing "completely independent variables are flagged as independent"
    (let [df (independent-data 1000)]
      (is (independent? df :y :x)
          "x and y are unrelated draws"))))

;; ---------------------------------------------------------------------------
;; Tests: conditional independence / dependence
;; ---------------------------------------------------------------------------

(deftest independent?-conditional-fork
  (testing "fork: children are conditionally independent given parent"
    (let [df (fork-data 1000)]
      (is (independent? df :x :y :z)
          "given Z, x = Z+noise and y = Z+noise share no information"))))

(deftest dependent?-conditional-collider
  (testing "collider: parents are conditionally dependent given child (explaining away)"
    (let [df (collider-data 1000)]
      (is (dependent? df :x :y :z)
          "given Z = X+Y, knowing X determines Y exactly"))))

(deftest dependent?-marginal-fork
  (testing "fork: children are marginally dependent (share common cause)"
    (let [df (fork-data 1000)]
      (is (dependent? df :x :y)
          "x and y are correlated through Z"))))

(deftest independent?-marginal-collider
  (testing "collider: parents are marginally independent"
    (let [df (collider-data 1000)]
      (is (independent? df :x :y)
          "x and y are independent draws in the collider"))))
