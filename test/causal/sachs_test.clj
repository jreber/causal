(ns causal.sachs-test
  "Integration tests against the Sachs et al. (2005) protein signalling dataset.

  Tests that FCIT correctly identifies direct edges from the ground-truth DAG
  as statistically dependent.  This is the canonical benchmark for causal
  discovery algorithms.

  Ground-truth DAG (20 edges, 11 nodes — Sachs et al., Science 2005):

    PIP3 → Plcg, PIP3 → PIP2, PIP3 → Akt, PIP3 → PKA (bidirectional)
    Plcg → PIP2, Plcg → PKC
    PIP2 → PKC
    PKC → PKA (bidirectional via PIP3), PKC → Raf, PKC → Mek, PKC → Jnk, PKC → P38
    PKA → Raf, PKA → Mek, PKA → Erk, PKA → Akt, PKA → Jnk, PKA → P38
    Raf → Mek
    Mek → Erk

  The tests cover two structural claims:
    1. Marginal dependence — each directed edge A → B implies A ⊥̸ B.
    2. Conditional independence — blocking all active paths via a conditioning
       set implies d-separation.  Because every pair of non-adjacent nodes is
       connected via multiple paths in this dense graph, conditioning sets
       typically require 2+ variables.  The multi-variable FCIT API
       (dependent? ds target predictor cond1 cond2 …) supports this directly.

  Data downloaded on demand from:
    https://zenodo.org/records/7681811/files/sachs.zip?download=1"
  (:require [clojure.test :refer :all]
            [tech.v3.dataset :as tmd]
            [causal.fcit :refer [dependent? independent?]]
            [causal.causal :refer [fork-violations fork?
                                   chain-violations chain?
                                   collider-violations collider?]]
            [clojure.java.io :as io])
  (:import [java.util.zip ZipInputStream]
           [java.net URL]))

;; ---------------------------------------------------------------------------
;; Data loading
;; ---------------------------------------------------------------------------

(def ^:private sachs-zip-url
  "https://zenodo.org/records/7681811/files/sachs.zip?download=1")

(def ^:private cache-path
  (str (System/getProperty "java.io.tmpdir") "/sachs_cd3cd28.csv"))

(defn- download-and-extract!
  "Downloads sachs.zip from Zenodo and extracts the observational CSV
  (cd3cd28.csv) to a local temp file.  No-ops when the cache already exists."
  []
  (when-not (.exists (io/file cache-path))
    (with-open [zip-in (ZipInputStream. (.openStream (URL. sachs-zip-url)))]
      (loop []
        (when-let [entry (.getNextEntry zip-in)]
          (if (.endsWith (.getName entry) "cd3cd28.csv")
            (with-open [out (io/output-stream cache-path)]
              (io/copy zip-in out))
            (do (.closeEntry zip-in)
                (recur))))))))

(defonce ^:private sachs-ds
  (do
    (download-and-extract!)
    ;; key-fn converts string column names ("Raf", "Mek", …) to keywords.
    (tmd/->dataset cache-path {:key-fn keyword})))

;; ---------------------------------------------------------------------------
;; Tests: all 18 ground-truth edges should show as statistically dependent
;;
;; We test every edge A → B from the Sachs ground-truth DAG except the
;; bidirectional PIP3 ↔ PKA pair (a feedback cycle that the DAG formalism
;; doesn't cleanly handle).
;;
;; Note: some pairs below share common causes (e.g. PKA and PKC both drive
;; Jnk, Raf, Mek, P38).  The marginal test asserts statistical dependence via
;; *any* active path — direct edge, indirect effect, or shared parent — which
;; is the correct claim when we know a direct edge exists.
;; ---------------------------------------------------------------------------

;; MAPK cascade
(deftest ^:sachs raf-and-mek-are-dependent
  (testing "Raf → Mek"
    (is (dependent? sachs-ds :Mek :Raf))))

(deftest ^:sachs mek-and-erk-are-dependent
  (testing "Mek → Erk"
    (is (dependent? sachs-ds :Erk :Mek))))

;; PKA edges
(deftest ^:sachs pka-and-raf-are-dependent
  (testing "PKA → Raf"
    (is (dependent? sachs-ds :Raf :PKA))))

(deftest ^:sachs pka-and-mek-are-dependent
  (testing "PKA → Mek"
    (is (dependent? sachs-ds :Mek :PKA))))

(deftest ^:sachs pka-and-erk-are-dependent
  (testing "PKA → Erk"
    (is (dependent? sachs-ds :Erk :PKA))))

(deftest ^:sachs pka-and-akt-are-dependent
  (testing "PKA → Akt"
    (is (dependent? sachs-ds :Akt :PKA))))

(deftest ^:sachs pka-and-jnk-are-dependent
  (testing "PKA → Jnk"
    (is (dependent? sachs-ds :Jnk :PKA))))

(deftest ^:sachs pka-and-p38-are-dependent
  (testing "PKA → P38"
    (is (dependent? sachs-ds :P38 :PKA))))

;; PKC edges
(deftest ^:sachs pkc-and-raf-are-dependent
  (testing "PKC → Raf"
    (is (dependent? sachs-ds :Raf :PKC))))

(deftest ^:sachs pkc-and-mek-are-dependent
  (testing "PKC → Mek"
    (is (dependent? sachs-ds :Mek :PKC))))

(deftest ^:sachs pkc-and-jnk-are-dependent
  (testing "PKC → Jnk"
    (is (dependent? sachs-ds :Jnk :PKC))))

(deftest ^:sachs pkc-and-p38-are-dependent
  (testing "PKC → P38"
    (is (dependent? sachs-ds :P38 :PKC))))

;; PIP lipid signalling
(deftest ^:sachs pip3-and-plcg-are-dependent
  (testing "PIP3 → Plcg"
    (is (dependent? sachs-ds :Plcg :PIP3))))

(deftest ^:sachs pip3-and-pip2-are-dependent
  (testing "PIP3 → PIP2"
    (is (dependent? sachs-ds :PIP2 :PIP3))))

(deftest ^:sachs pip3-and-akt-are-dependent
  (testing "PIP3 → Akt"
    (is (dependent? sachs-ds :Akt :PIP3))))

(deftest ^:sachs plcg-and-pip2-are-dependent
  (testing "Plcg → PIP2"
    (is (dependent? sachs-ds :PIP2 :Plcg))))

(deftest ^:sachs plcg-and-pkc-are-dependent
  (testing "Plcg → PKC"
    (is (dependent? sachs-ds :PKC :Plcg))))

(deftest ^:sachs pip2-and-pkc-are-dependent
  (testing "PIP2 → PKC"
    (is (dependent? sachs-ds :PKC :PIP2))))

;; ---------------------------------------------------------------------------
;; Tests: conditional independence via multi-variable d-separation
;;
;; All active paths between these pairs pass through the listed conditioning
;; set, so conditioning should render them independent.
;; ---------------------------------------------------------------------------

(deftest ^:sachs jnk-and-p38-independent-given-pka-pkc
  (testing "Jnk ⊥ P38 | {PKA, PKC}: both are pure downstream effects of the same two regulators"
    ;; Active paths: Jnk ← PKA → P38  and  Jnk ← PKC → P38
    ;; Both are blocked by conditioning on {PKA, PKC}.
    (is (independent? sachs-ds :Jnk :P38 :PKA :PKC)
        "No path from Jnk to P38 that avoids both PKA and PKC")))

(deftest ^:sachs raf-and-p38-independent-given-pka-pkc
  (testing "Raf ⊥ P38 | {PKA, PKC}: Raf and P38 share only PKA and PKC as regulators"
    ;; Active paths: Raf ← PKA → P38  and  Raf ← PKC → P38
    ;; Both are blocked by conditioning on {PKA, PKC}.
    (is (independent? sachs-ds :Raf :P38 :PKA :PKC)
        "No path from Raf to P38 that avoids both PKA and PKC")))

(deftest ^:sachs mek-and-akt-independent-given-pka
  (testing "Mek ⊥ Akt | PKA: all paths between Mek and Akt funnel through PKA"
    ;; Active paths from Mek to Akt:
    ;;   Mek ← PKA → Akt               (blocked by PKA ✓)
    ;;   Mek ← Raf ← PKA → Akt         (blocked by PKA ✓)
    ;;   Mek ← PKA ← PIP3 → Akt        (blocked by PKA ✓)
    ;; PKC has no direct path to Akt (PKC → Mek/Raf/Jnk/P38 only).
    (is (independent? sachs-ds :Mek :Akt :PKA)
        "PKA is the sole bridge between the Mek and Akt sub-networks")))

;; ---------------------------------------------------------------------------
;; Tests: causal structure detection (fork / chain / collider)
;;
;; Uses the fork?, chain?, collider? helpers from causal.causal, extended to
;; support multi-variable conditioning sets.
;; ---------------------------------------------------------------------------

(deftest ^:sachs sachs-fork-structure
  (testing "Fork: Jnk ← {PKA, PKC} → P38
  Two distinct overlapping forks sharing the same children.  Both PKA and PKC
  are required in the conditioning set to d-separate Jnk and P38."
    (is (fork? sachs-ds {:fork/parents [:PKA :PKC]
                         :fork/child1  :Jnk
                         :fork/child2  :P38}))))

(deftest ^:sachs sachs-chain-structure
  (testing "Chain: Raf → Mek → Erk, with PKA as confounder
  PKA drives both Raf and Erk directly, creating a backdoor path that must be
  blocked alongside the chain middle node Mek."
    (is (chain? sachs-ds {:chain/first       :Raf
                          :chain/middle      :Mek
                          :chain/last        :Erk
                          :chain/confounders [:PKA]}))))

(deftest ^:sachs sachs-collider-structure
  (testing "Collider: PKA → Jnk ← PKC, decoupled via PIP3
  PKA and PKC are d-separated by conditioning on PIP3 (all paths between them
  pass through PIP3).  Conditioning on the collider Jnk then re-opens the
  blocked path, demonstrating the explaining-away effect."
    (is (collider? sachs-ds {:collider/parent1    :PKA
                             :collider/parent2    :PKC
                             :collider/child      :Jnk
                             :collider/decouplers [:PIP3]}))))
