-- VERITAS Verifier: Root module
-- Building this forces all Mathlib oleans into the cache,
-- so subsequent per-theorem lean calls take 1-5s instead of 60s+.
import Mathlib
import Aesop

open BigOperators Real Nat Rat Finset
