(ns causal.time-series-paper-test
  (:require [clojure.test :refer :all]
            [causal.time-series-paper :refer :all]
            [tablecloth.api :as tc]
            [tablecloth.column.api :as tcc]
            [scicloj.ml.dataset :as ds]
            [tech.v3.dataset.column :as dsc]
            [ubergraph.core :as ug]
            [ubergraph.alg :as ug-algo]))

(defn gen-rand [& _]
  (-> (rand)
      (- 0.5)
      (* 10)))

(defn noisy-identity [x] (+ x (gen-rand)))

(defn gen-independent-col []
  (tcc/column (repeatedly 500 gen-rand))
  #_(tcc/column (range 100)))

(defn gen-independent-ds []
  (tc/dataset {:a (gen-independent-col)}))

(defn gen-depdendent-ds [ws f]
  (let [num-xs (count ws)
        original-ds (-> (keys ws)
                        (zipmap (repeatedly num-xs #(gen-independent-col)))
                        tc/dataset)
        y (-> (reduce-kv
               (fn [ds col lag]
                 (tc/shift ds col col (- lag)))
               original-ds
               ws)
              (tc/map-columns :y :all f)
              (get :y))]
    (tc/add-column original-ds :y y)))

(defn gen-dependent-ds-v2
  [graph timesteps]
  (let [nodes (ug/nodes graph)
        data (atom (zipmap nodes (repeat [])))]
    (doseq [t (range timesteps)]
      (doseq [node (ug-algo/topsort graph)]
        (let [parent-edges (ug/in-edges graph node)
              lagged-values (mapv
                             (fn [edge]
                               (let [parent (ug/src edge)
                                     lag (:lag (ug/attrs graph edge))
                                     parent-series (@data parent)]
                                 (if (>= (count parent-series) (+ lag 1))
                                   (nth parent-series (- (count parent-series) lag 1))
                                   0))) ; Default to 0 if not enough history
                             parent-edges)
              combiner (:combine-fn (ug/attrs graph node) (fn [& _] 0)) ; Default function
              node-value (apply combiner lagged-values)]
          (swap! data update node conj node-value))))
    (-> @data
        tc/dataset
        (tc/update-columns :all reverse))))

(deftest shift-test
  (testing "[Columns]"
    (testing "Positive shift"
      (let [col (tcc/column [0 1 2 3 4])]
        (is (= (tcc/column [1 2 3 4 nil])
               (shift col 1)))))
    (testing "Negative shift"
      (let [col (tcc/column [0 1 2 3 4])]
        (is (= (tcc/column [nil 0 1 2 3])
               (shift col -1)))))
    (testing "With Time-Series Generator"
      (let [num-observations 5
            ds (-> (ug/digraph)
                   (ug/add-nodes-with-attrs [:A {:combine-fn gen-rand}]
                                            [:B {:combine-fn identity}])
                   (ug/add-edges [:A :B {:lag 1}])
                   (gen-dependent-ds-v2 num-observations))
            A (get ds :A)
            B (get ds :B)]
        (is (= (tcc/slice B           :start (- num-observations 2))
               (tcc/slice (shift A 1) :start (- num-observations 2)))))))

  (testing "[Datasets]"
    (testing "Positive shift"
      (let [ds (tc/dataset {:a [0 1 2 3 4]
                            :b [0 1 2 3 4]})]
        (is (= (tc/dataset {:a [1 2 3 4 nil]
                            :b [1 2 3 4 nil]})
               (shift ds 1)))))
    (testing "Negative shift"
      (let [ds (tc/dataset {:a [0 1 2 3 4]
                            :b [0 1 2 3 4]})]
        (is (= (tc/dataset {:a [nil 0 1 2 3]
                            :b [nil 0 1 2 3]})
               (shift ds -1)))))))

(deftest min-lag-test
  (testing "X and Y are time-shifted versions of each other"
    (let [w 1
          x (gen-independent-col)
          y (shift x w)]
      (is (= w (min-lag y x)))))
  (testing "X and Y are time-shifted versions of each other (bigger w)"
    (let [w 3
          x (gen-independent-col)
          y (shift x w)]
      (is (= w (min-lag y x)))))
  (testing "Simple single-lag cause"
    (let [ds (-> (ug/digraph)
                 (ug/add-nodes-with-attrs [:x0 {:combine-fn gen-rand}]
                                          [:y {:combine-fn noisy-identity}])
                 (ug/add-edges [:x0 :y {:lag 1}])
                 (gen-dependent-ds-v2 500))]
      (is (= 1 (min-lag ds :y :x0)))))
  (testing "Two causes at different lags"
    (let [ds (-> (ug/digraph)
                 (ug/add-nodes-with-attrs [:x0 {:combine-fn gen-rand}]
                                          [:x1 {:combine-fn gen-rand}]
                                          [:y {:combine-fn +}])
                 (ug/add-edges [:x0 :y {:lag 1}]
                               [:x1 :y {:lag 2}])
                 (gen-dependent-ds-v2 500))]
      (is (= 2 (min-lag ds :y :x1)))))
  (testing "A->B->C, each edge is lag=1; lag A->C should be 2"
    (let [ds (-> (ug/digraph)
                 (ug/add-nodes-with-attrs [:A {:combine-fn gen-rand}]
                                          [:B {:combine-fn noisy-identity}]
                                          [:C {:combine-fn noisy-identity}])
                 (ug/add-edges [:A :B {:lag 1}]
                               [:B :C {:lag 1}])
                 (gen-dependent-ds-v2 500))]
      (is (= 2 (min-lag ds :C :A)))
      (is (= 1 (min-lag ds :C :B)))))
  (testing "A->C<-B"
    (let [ds (-> (ug/digraph)
                 (ug/add-nodes-with-attrs [:A {:combine-fn gen-rand}]
                                          [:B {:combine-fn gen-rand}]
                                          [:C {:combine-fn +}])
                 (ug/add-edges [:A :C {:lag 1}]
                               [:B :C {:lag 2}])
                 (gen-dependent-ds-v2 1000))]
      (is (= 1 (min-lag ds :C :A)))
      (is (= 2 (min-lag ds :C :B))))))

(deftest ->ds-test
  (testing "Mix of columns and datasets"
    (let [col1 (gen-independent-col)
          ds (gen-independent-ds)
          col2 (gen-independent-col)]
      (is (= (tc/dataset (concat [col1]
                                 (tc/columns ds)
                                 [col2]))
             (->ds col1 ds col2))))))

(deftest ->dependent-args-test
    ;; doesn't work; names of anonymous cols are inconsistent, and ordering of cols is inconsistent
    ;; would pass, though, if I could resolve those
  #_(testing "Basic example"
      (let [targets (gen-depdendent-ds {:x 1} +)
            conditioning-col (gen-independent-col)
            conditioning-ds (gen-independent-ds)]
        (is (= [(->ds targets conditioning-col conditioning-ds)
                :y :x0 0 :a]
               (->dependent-args (get targets :y)
                                 (get targets :x)
                                 conditioning-col
                                 conditioning-ds))))))

(deftest conditioning-set-test
  (testing "[Single Lag]"
    (testing "One variable present"
      (let [ds (tc/dataset {:A (gen-independent-col)})
            w (constantly 1)]
        (is (= (tc/dataset)
               (conditioning-set ds w (get ds :A))))))
    (testing "Two variables present"
      (let [ds (tc/dataset {:A [0 1 2 3 4]
                            :B [0 1 2 3 4]})
            w (constantly 1)]
        (is (= (-> ds
                   (tc/drop-columns :A)
                   (shift -1))
               (conditioning-set ds w (get ds :A)))))))
  (testing "[Multi Lag]"
    (testing "Two variables present"
      (let [ds (tc/dataset {:A [0 1 2 3 4]
                            :B [5 6 7 8 9]})
            w {(get ds :A) 1
               (get ds :B) 2}]
        (= (-> ds
               (tc/drop-columns :A)
               (shift -2))
           (conditioning-set ds w (get ds :A)))))))

(deftest dependence-test
  (testing "[Marginally Dependent]"
    (testing "Very simple"
      (let [col (gen-independent-col)]
        (is (dependent? col col))))
    (testing "Simple causal"
      (let [ds (-> (ug/digraph)
                   (ug/add-nodes-with-attrs [:A {:combine-fn gen-rand}]
                                            [:B {:combine-fn noisy-identity}])
                   (ug/add-edges [:A :B {:lag 1}])
                   (gen-dependent-ds-v2 1000))]
        (is (dependent? (get ds :B)
                        (shift (get ds :A) 1))))))
  (testing "[Marginally Independent]"
    (testing "Very simple"
      (let [A (gen-independent-col)
            B (gen-independent-col)]
        (is (independent? A B)))))
  (testing "[Conditionally Independent"
    (testing "Simple Fork"
      (let [ds (-> (ug/digraph)
                   (ug/add-nodes-with-attrs [:A {:combine-fn gen-rand}]
                                            [:B {:combine-fn noisy-identity}]
                                            [:C {:combine-fn noisy-identity}])
                   (ug/add-edges [:A :B {:lag 1}])
                   (ug/add-edges [:A :C {:lag 1}])
                   (gen-dependent-ds-v2 500))
            A (get ds :A)
            B (get ds :B)
            C (get ds :C)]
        (is (dependent? B C))
        (is (independent? B C (shift A 1))))))
  (testing "[Conditionally Dependent"
    (testing "Simple Collider"
      (let [ds (-> (ug/digraph)
                   (ug/add-nodes-with-attrs [:A {:combine-fn gen-rand}]
                                            [:B {:combine-fn gen-rand}]
                                            [:C {:combine-fn +}])
                   (ug/add-edges [:A :C {:lag 1}])
                   (ug/add-edges [:B :C {:lag 1}])
                   (gen-dependent-ds-v2 500))
            A (get ds :A)
            B (get ds :B)
            C (get ds :C)]
        (is (independent? A B))
        (is (dependent? A B (shift C -1)))))))

(deftest sypi-test
  (testing "[Single Lag=1]"
    (testing "Basic Collider A->C<-B"
      (let [ds (-> (ug/digraph)
                   (ug/add-nodes-with-attrs [:A {:combine-fn gen-rand}]
                                            [:B {:combine-fn gen-rand}]
                                            [:C {:combine-fn +}])
                   (ug/add-edges [:A :C {:lag 1}]
                                 [:B :C {:lag 2}])
                   (gen-dependent-ds-v2 1000))]
        (is (= #{:A :B}
               (find-sg-unconfounded-v2 ds :C)))))
    (testing "Basic Fork B<-A->C"
      (let [ds (-> (ug/digraph)
                   (ug/add-nodes-with-attrs [:A {:combine-fn gen-rand}]
                                            [:B {:combine-fn noisy-identity}]
                                            [:C {:combine-fn noisy-identity}])
                   (ug/add-edges [:A :B {:lag 1}]
                                 [:A :C {:lag 1}])
                   (gen-dependent-ds-v2 1000))]
        (is (empty?
             (find-sg-unconfounded-v2 ds :A)))))
    (testing "Basic Chain A->B->C"
      (let [ds (-> (ug/digraph)
                   (ug/add-nodes-with-attrs [:A {:combine-fn gen-rand}]
                                            [:B {:combine-fn noisy-identity}]
                                            [:C {:combine-fn noisy-identity}])
                   (ug/add-edges [:A :B {:lag 1}]
                                 [:B :C {:lag 1}])
                   (gen-dependent-ds-v2 1000))]
        (is (= #{:A}
               (find-sg-unconfounded-v2 ds :B)))
        (is (= #{:A :B}
               (find-sg-unconfounded-v2 ds :C)))))
    (testing "Basic Diamond"
      (let [ds (-> (ug/digraph)
                   (ug/add-nodes-with-attrs [:A {:combine-fn gen-rand}]
                                            [:B {:combine-fn noisy-identity}]
                                            [:C {:combine-fn noisy-identity}]
                                            [:D {:combine-fn +}])
                   (ug/add-edges [:A :B {:lag 1}]
                                 [:A :C {:lag 1}]
                                 [:B :D {:lag 1}]
                                 [:C :D {:lag 1}])
                   (gen-dependent-ds-v2 1000))]
        (is (= #{:A :B :C}
               (find-sg-unconfounded-v2 ds :D)))))))
