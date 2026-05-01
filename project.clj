(defproject causal "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [[org.clojure/clojure "1.12.4"]
                 [com.taoensso/encore "3.160.1"]
                 [scicloj/scicloj.ml "0.3"]
                 [scicloj/tablecloth "7.029.2"]
                 [techascent/tech.ml.dataset "7.032"]
                 [metasoarous/oz "1.6.0-alpha36"]
                 [generateme/fastmath "2.4.0"]
                 [ubergraph "0.9.0"]]
  :main ^:skip-aot causal.core
  :target-path "target/%s"
  :jvm-opts ["-Xmx16g" "-XX:-OmitStackTraceInFastThrow"]
  :test-selectors {:default (complement :sachs)
                   :sachs   :sachs
                   :all     (constantly true)}
  :profiles {:uberjar {:aot :all
                       :jvm-opts ["-Dclojure.compiler.direct-linking=true"]}})
