(ns causal.causal-test
  "Integration tests for causal structure detection (fork, chain, collider).

  Each test verifies both that the correct structure is detected AND that
  violations are empty (no false failures). These tests run ~12 ML models
  each, so expect ~30s per test."
  (:require [clojure.test :refer :all]
            [scicloj.ml.dataset :as ds]
            [causal.causal :refer [fork-violations fork?
                                   chain-violations chain?
                                   collider-violations collider?
                                   make-fork-df make-chain-df make-collider-df]]))

;; ---------------------------------------------------------------------------
;; Additional synthetic data generators
;; (make-fork-df / make-chain-df / make-collider-df cover the simple cases;
;;  these cover the extended API: multiple parents, confounders, decouplers)
;; ---------------------------------------------------------------------------

(defn- two-parent-fork-data
  "W and Z each independently cause both X and Y.
  X ⊥ Y | {W, Z} but X and Y are marginally dependent."
  [n]
  (let [w (map #(- % 0.5) (repeatedly n rand))
        z (map #(- % 0.5) (repeatedly n rand))
        x (map + w z (map #(* 0.5 (- % 0.5)) (repeatedly n rand)))
        y (map + w z (map #(* 0.5 (- % 0.5)) (repeatedly n rand)))]
    (ds/dataset {:x x :y y :w w :z z})))

(defn- confounded-chain-data
  "Chain A → B → C plus confounder W → A and W → C.
  A ⊥ C | {B, W} but A ⊥̸ C | B alone (backdoor path A ← W → C)."
  [n]
  (let [w (map #(- % 0.5) (repeatedly n rand))
        a (map + w (map #(- % 0.5) (repeatedly n rand)))
        b (map + a (map #(- % 0.5) (repeatedly n rand)))
        c (map + b w (map #(- % 0.5) (repeatedly n rand)))]
    (ds/dataset {:a a :b b :c c :w w})))

(defn- decoupled-collider-data
  "Collider P1 → C ← P2, where P1 and P2 share common cause D.
  P1 ⊥ P2 | D (decoupler blocks), but P1 ⊥̸ P2 | {D, C} (collider opens)."
  [n]
  (let [d  (map #(- % 0.5) (repeatedly n rand))
        p1 (map + d (map #(- % 0.5) (repeatedly n rand)))
        p2 (map + d (map #(- % 0.5) (repeatedly n rand)))
        c  (map + p1 p2)]
    (ds/dataset {:d d :p1 p1 :p2 p2 :c c})))

;; ---------------------------------------------------------------------------
;; Fork: Z -> X, Z -> Y  (single parent, backward-compatible)
;; ---------------------------------------------------------------------------

(deftest fork-has-no-violations
  (testing "fork structure X<-Z->Y satisfies all d-separation criteria"
    (let [df (make-fork-df 1000)
          violations (fork-violations df {:fork/parent  :z
                                          :fork/child1  :x
                                          :fork/child2  :y})]
      (is (empty? violations)
          (str "Unexpected violations: " violations)))))

(deftest fork?-returns-true-for-fork-data
  (testing "fork? predicate recognises a fork structure"
    (let [df (make-fork-df 1000)]
      (is (fork? df {:fork/parent :z :fork/child1 :x :fork/child2 :y})))))

;; ---------------------------------------------------------------------------
;; Fork: extended API — multiple parents, single-keyword :fork/parents
;; ---------------------------------------------------------------------------

(deftest fork?-detects-two-parent-fork
  (testing ":fork/parents vector: X ← {W, Z} → Y detected as fork"
    (let [df (two-parent-fork-data 1000)]
      (is (fork? df {:fork/parents [:w :z] :fork/child1 :x :fork/child2 :y})))))

(deftest fork?-accepts-single-keyword-for-parents
  (testing ":fork/parents accepts a bare keyword (not only a vector)"
    (let [df (make-fork-df 1000)]
      (is (fork? df {:fork/parents :z :fork/child1 :x :fork/child2 :y})))))

(deftest fork-violations-throws-without-parent-spec
  (testing "fork-violations throws ex-info when neither :fork/parent nor :fork/parents is given"
    (is (thrown? clojure.lang.ExceptionInfo
                 (fork-violations (make-fork-df 10) {:fork/child1 :x :fork/child2 :y})))))

;; ---------------------------------------------------------------------------
;; Chain: X -> Y -> Z  (no confounders, backward-compatible)
;; ---------------------------------------------------------------------------

(deftest chain-has-no-violations
  (testing "chain structure X->Y->Z satisfies all d-separation criteria"
    (let [df (make-chain-df 1000)
          violations (chain-violations df {:chain/first  :x
                                           :chain/middle :y
                                           :chain/last   :z})]
      (is (empty? violations)
          (str "Unexpected violations: " violations)))))

(deftest chain?-returns-true-for-chain-data
  (testing "chain? predicate recognises a chain structure"
    (let [df (make-chain-df 1000)]
      (is (chain? df {:chain/first :x :chain/middle :y :chain/last :z})))))

;; ---------------------------------------------------------------------------
;; Chain: extended API — confounders as vector and as single keyword
;; ---------------------------------------------------------------------------

(deftest chain?-detects-confounded-chain
  (testing ":chain/confounders vector: A→B→C with confounder W detected as chain"
    (let [df (confounded-chain-data 1000)]
      (is (chain? df {:chain/first       :a
                      :chain/middle      :b
                      :chain/last        :c
                      :chain/confounders [:w]})))))

(deftest chain?-accepts-single-keyword-for-confounders
  (testing ":chain/confounders accepts a bare keyword (not only a vector)"
    (let [df (confounded-chain-data 1000)]
      (is (chain? df {:chain/first       :a
                      :chain/middle      :b
                      :chain/last        :c
                      :chain/confounders :w})))))

;; ---------------------------------------------------------------------------
;; Collider: X -> Z <- Y  (no decouplers, backward-compatible)
;; ---------------------------------------------------------------------------

(deftest collider-has-no-violations
  (testing "collider structure X->Z<-Y satisfies all d-separation criteria"
    (let [df (make-collider-df 1000)
          violations (collider-violations df {:collider/parent1 :x
                                              :collider/parent2 :y
                                              :collider/child   :z})]
      (is (empty? violations)
          (str "Unexpected violations: " violations)))))

(deftest collider?-returns-true-for-collider-data
  (testing "collider? predicate recognises a collider structure"
    (let [df (make-collider-df 1000)]
      (is (collider? df {:collider/parent1 :x
                         :collider/parent2 :y
                         :collider/child   :z})))))

;; ---------------------------------------------------------------------------
;; Collider: extended API — decouplers
;; ---------------------------------------------------------------------------

(deftest collider?-detects-decoupled-collider
  (testing ":collider/decouplers: P1→C←P2 with parents connected via D detected as collider"
    (let [df (decoupled-collider-data 1000)]
      (is (collider? df {:collider/parent1    :p1
                         :collider/parent2    :p2
                         :collider/child      :c
                         :collider/decouplers [:d]})))))

;; ---------------------------------------------------------------------------
;; Negative tests: wrong structure is rejected
;;
;; Note: fork and chain are Markov-equivalent (same d-separation patterns),
;; so fork? on chain data and chain? on fork data both return true.
;; The distinguishable case is the collider, which has opposite marginal
;; independence structure.
;; ---------------------------------------------------------------------------

(deftest fork?-returns-false-for-collider-data
  (testing "fork? rejects collider data: children are marginally independent, not dependent"
    (let [df (make-collider-df 1000)]
      (is (not (fork? df {:fork/parent :z :fork/child1 :x :fork/child2 :y}))))))

(deftest chain?-returns-false-for-collider-data
  (testing "chain? rejects collider data: endpoints are marginally independent, not dependent"
    (let [df (make-collider-df 1000)]
      (is (not (chain? df {:chain/first :x :chain/middle :z :chain/last :y}))))))

(deftest collider?-returns-false-for-fork-data
  (testing "collider? rejects fork data: parents are marginally dependent, not independent"
    (let [df (make-fork-df 1000)]
      (is (not (collider? df {:collider/parent1 :x :collider/parent2 :y :collider/child :z}))))))

(deftest collider?-returns-false-for-chain-data
  (testing "collider? rejects chain data: endpoints are marginally dependent, not independent"
    (let [df (make-chain-df 1000)]
      (is (not (collider? df {:collider/parent1 :x :collider/parent2 :z :collider/child :y}))))))
