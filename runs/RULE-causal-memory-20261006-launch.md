# Authorized bounded causal pilot · 5 October 2026

Oscar authorized up to five total GPU hours from existing free quota, no submission. No paid use.
Run the previously sealed private control and treatment exactly once, each with Kaggle's server-side
session_timeout_seconds=8400 (2h20m), total maximum requested session execution=16800s (4h40m).
Both use NvidiaRtxPro6000, oneGPU per session;20minutes headroom remains below5hours. No retries,
new arms, scoring submissions or continuation. Platform enforcement is the primary cutoff;
local polling cannot enforce it if the laptop sleeps. A forced timeout means INVALID under the
existing25game denominator; never exclude unfinished games or report synthetic scores.

Frozen notebooks, baseline and runtime settings are unchanged. Fresh external observations are
in the adjacent receipt's snapshots directory. Every file enumeration was paginated fully:
probe7, runtime7, Duck4408, competition148. Model version1 id977939 remains the only version.
Downloaded probe package content hash and runtime manifest hash exactly match the frozen seal.
Old API-list hash canonicalization was not recorded; therefore no exact comparison of those
old list hashes is asserted. The new raw API lists are sealed before launch as an explicit
observation amendment, not silently substituted for the historical content proof. Duck file
creation dates remain31August; model/runtime source identities are independently checked by
observed runtime receipts after execution. Mismatch remains INVALID, not permission to alter rules.

Stop after these two attempts irrespective of findings. Historical5GiBKV config is retained,
not the known7/8GiBOOM arms. Exact notebook hashes and response runversions are in launch.json.
The supervisor saves periodic live status/quota and pulls terminal outputs. Timeout acceptance is
based on the installed Kaggle API sending session_timeout_seconds, not a local timer promise.
