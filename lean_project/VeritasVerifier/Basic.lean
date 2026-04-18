-- VERITAS Verifier: Base module
-- Building this forces Mathlib oleans to be cached,
-- so all subsequent per-theorem validations are fast (1-5s instead of 60s+).
import Mathlib
import Aesop

open BigOperators Real Nat Rat
