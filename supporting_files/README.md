# Interactive UKB Dataset Visualization

Open `index.html` in a browser. The HTML is self-contained for interactive viewing and includes aggregate coverage plus two non-ID example values per sampled feature group for internal lab use.

## Moving or copying

You can copy this whole `interactive_dataset_visualization` folder to another drive or computer and open the outer `index.html` there. The app does not depend on local absolute paths. Support files live in `supporting_files/`; keep that folder if you want the exported CSV/JSON tables, static SVG figures, and this README. Tree audit/source files live in `source_files/`. The HTML itself embeds the data needed for the browser view, so moving support/source files does not change the visualization. The only absolute links are external UKB Showcase web URLs used as references.

## Key outputs

- `index.html`: interactive report appendix.
- `semantic_feature_groups.csv`: cognitive feature groups with raw-column traceability.
- `semantic_group_coverage.csv`: exact group coverage overall and by UKB visit instance.
- `semantic_node_coverage.csv`: parent-node coverage for the tree view.
- `tree_structure_review_candidates.csv`: deterministic review queue for tree/search taxonomy corrections.
- `tree_structure_audit_summary.csv`: tree/search audit metrics and biological-domain counts.
- `semantic_feature_tree.json`: data payload used by the HTML.
- `target_catalog.csv`: exact target candidates and phenotype groups.
- `target_predictor_availability.csv`: metadata-derived predictor availability per target.
- `dataset_alignment_bridge_summary.csv`: bridge-file uniqueness checks and local-source overlaps.
- `dataset_alignment_set_summary.csv`: bridge-aware aligned participant counts by source.
- `id_namespace_interpretation.csv`: report-facing ID namespace and bridge interpretation.
- `profile_participant_groups.csv`, `profile_disease_targets.csv`, `profile_missingness_patterns.csv`: aggregate Basic Profiling tab inputs. Feature distribution and repeatability CSVs are generated as side outputs but not shown in the app.
- `dataset_tree_overview.svg`, `storage_pattern_explanation.svg`, `prediction_target_overview.svg`, `missing_requested_data_types.svg`: static report figures.
- `validation_report.json`: mapping and paired-array validation checks.

For tree review, use `../source_files/` from this README location. It contains copied inputs, generated semantic outputs, coverage tables, audit files, and the generator script snapshot.

Validation status: `PASS`; mapped raw columns: 22,530.
