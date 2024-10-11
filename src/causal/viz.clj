(ns causal.viz
  "Helpers for visualizing data."
  (:require [tech.v3.dataset :as ds]
            [tech.v3.datatype.functional :as dfn]
            [oz.core :as oz]))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Generic display helpers
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def start-oz-server
  "Starts an Oz server for use in a REPL session."
  oz/start-server!)

(def view! oz/view!)

(defn view-adjacent! [& charts]
  "Displays the given Vega charts side-by-side."
  (let [hiccup [:div {:style {:display "flex" :flex-direction "row"}}
                (for [chart charts]
                  [:vega-lite chart])]]
    (view! hiccup)))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Helpers for creating specific visualizations
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


(defn histogram
  "Creates a histogram from the given information.

  Can either be:
  - provided a `tech.ml.dataset`, a column name, and an
  alias for that column"
  ([ds col alias]
   (histogram
    (map #(into {} %)
         (-> ds
             (ds/select-columns [col])
             (ds/rename-columns {col alias})
             ds/mapseq-reader))
    alias))
  ([data col]
   {:width 600
    :height 600
    :data {:name (str "histogram of " col)
           :values data}
    :mark {:type "bar"}
    :encoding {:x {:bin true
                   :field col
                   :type :quantitative}
               :y {:aggregate "count"
                   :type :quantitative
                   :scale {:type "log"}}}}))


