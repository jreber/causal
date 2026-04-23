(ns causal.causal-test
  "Integration tests for causal structure detection (fork, chain, collider).

  Each test verifies both that the correct structure is detected AND that
  violations are empty (no false failures). These tests run ~12 ML models
  each, so expect ~30s per test."
  (:require [clojure.test :refer :all]
            [causal.causal :refer [fork-violations fork?
                                   chain-violations chain?
                                   collider-violations collider?
                                   make-fork-df make-chain-df make-collider-df]]))

;; ---------------------------------------------------------------------------
;; Fork: Z -> X, Z -> Y
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
;; Chain: X -> Y -> Z
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
;; Collider: X -> Z <- Y
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
