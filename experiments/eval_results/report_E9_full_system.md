# MedFusion-RAFT Benchmark Evaluation Report
Generated: 2026-11-07 20:55:56 UTC

## Summary Table

| Experiment ID | Name | Accuracy | F1 | Claim Support | Selective Risk | Coverage | ECE | Median Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| `E9_full_system` | Complete System with Adaptive Routing | 0.000 | 0.047 | 0.000 | 1.000 | 0.000 | 0.000 | 3.3 |

## Methodology & Guardrails
- **Accuracy**: Exact & token containment matching against gold reference answers.
- **Claim Support Rate**: Ratio of extracted atomic propositions verified by evidence passages.
- **Selective Risk**: Error rate evaluated specifically over non-abstained predictions.
- **Expected Calibration Error (ECE)**: Agreement between model confidence and empirical accuracy.
