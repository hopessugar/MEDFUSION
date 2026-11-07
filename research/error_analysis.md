# MedFusion-RAFT Error Analysis Framework

## Error Taxonomy

### Generator Errors
| Error Type | Description | Detection Method |
|-----------|-------------|------------------|
| Hallucination | Generating facts not in evidence | Citation validation + NLI |
| Omission | Missing important information from evidence | Gold answer comparison |
| Misinterpretation | Incorrect understanding of medical context | Verification + manual review |
| Over-generalization | Making claims broader than evidence supports | Claim-level entailment |
| Incorrect citation | Citing wrong passage for a claim | Deterministic citation check |
| Format violation | Not following structured output format | Output parsing failure |

### Retrieval Errors
| Error Type | Description | Detection Method |
|-----------|-------------|------------------|
| Missed relevant | Gold passage not in top-K results | Recall@K measurement |
| Topical but irrelevant | Retrieved passage is on-topic but doesn't answer question | Cross-encoder reranking |
| Outdated evidence | Retrieved passage contains outdated information | Date metadata check |
| Wrong granularity | Retrieved chunk too broad or too narrow | Manual review |

### Verification Errors
| Error Type | Description | Detection Method |
|-----------|-------------|------------------|
| False acceptance | Incorrect answer verified as correct | Ground truth comparison |
| False rejection | Correct answer flagged as incorrect | Ground truth comparison |
| Claim splitting error | Incorrect atomic claim boundaries | Manual review |
| Citation false positive | Invalid citation marked as valid | Manual review |
| NLI threshold error | Entailment score misclassifies support | Threshold sensitivity analysis |

### Abstention Errors
| Error Type | Description | Detection Method |
|-----------|-------------|------------------|
| Under-abstention | Should have abstained but accepted | Correctness of accepted answers |
| Over-abstention | Should have accepted but abstained | Coverage metric |
| Miscalibration | Confidence scores don't match accuracy | ECE, reliability diagram |

## Error Analysis Procedure

1. **Collect**: Run pipeline on test set, save all intermediate results
2. **Categorize**: Classify each error by type from the taxonomy above
3. **Quantify**: Count error frequencies, compute rates per category
4. **Correlate**: Check if specific error types co-occur (e.g., retrieval failure → hallucination)
5. **Attribute**: Trace errors to their source component (retrieval, generator, verifier, calibration)
6. **Compare**: Compare error profiles across experiments (E1-E9)
7. **Prioritize**: Identify highest-impact error types for improvement

## Key Analysis Questions

1. What fraction of incorrect answers are caused by retrieval failures vs. generator errors?
2. Does verification catch the most harmful errors (contradictions, unsupported medical claims)?
3. What is the generator-verifier error correlation? (If high, independent verification is less valuable)
4. At what coverage level does selective risk become acceptable?
5. Which question types (specialty, difficulty, answerability) have highest error rates?
6. Does RAFT training reduce distractor-induced errors compared to SFT?

## Status: NOT YET EXECUTED
Error analysis results will be populated after experiments are run.

