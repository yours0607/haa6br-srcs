# Data Contract

No data are distributed with this repository. The only supported experiment input is a cleaned, analysis-ready directory with the logical identifier `haa6br_integrated_v1`.

## Directory layout

```text
haa6br_integrated_v1/
|-- data/
|   |-- us_ucmr4_core.csv
|   |-- us_ucmr4_enriched_strict.csv
|   |-- uk_dwi242_locked_external.csv
|   `-- uk_dbp2009_field_external.csv
`-- metadata/
    `-- model_feature_sets.json
```

The three US/metadata files are required for SRCS optimization and risk-budget reconstruction. `uk_dwi242_locked_external.csv` is additionally required unless `--skip-uk` is used. The historical benchmark requires all five files.

## Metadata contract

`metadata/model_feature_sets.json` must be a JSON object containing three arrays of column names:

- `transportable_core`
- `us_operational_core`
- `us_enriched_primary`

Their union must exclude the target-derived and audit-only fields listed in `run_new_experiments.FORBIDDEN`. The code stops if a forbidden field appears in a model feature set.

## Table contract

US core and enriched tables must contain:

- `sample_id`: stable sample identifier.
- `group_system_id`: water-system grouping identifier; it must not cross EPA-region folds.
- `sample_date`: parseable sampling date used to define chronological rounds.
- `epa_region`: numeric EPA region.
- `haa6br_ug_l`: HAA6Br outcome in micrograms per liter.
- `primary_analysis_eligible`: Boolean-like primary-cohort flag.
- Every feature named by the applicable arrays in `model_feature_sets.json`.

The historical runner also expects `is_2021_sensitivity` in `us_ucmr4_core.csv`. Optional `group_site_id` improves the eligibility audit; when absent, row counts are used instead.

`uk_dwi242_locked_external.csv` must contain `sample_id`, `group_system_id`, `sample_date`, `haa6br_ug_l`, and every `transportable_core` feature. `uk_dbp2009_field_external.csv` must contain `group_system_id`, `haa6br_ug_l`, and every `transportable_core` feature.

Dates with parsing failures are removed before chronological round assignment. Primary US rows with missing system IDs or targets are rejected. The package producer remains responsible for unit harmonization, censoring rules, identifier construction, and source-license compliance.

## US source

The US monitoring basis is EPA UCMR 4. EPA describes the UCMR 4 monitoring period as 2018-2020. The official occurrence-data page notes a March 2024 revision to the UCMR 4 occurrence data.

- EPA UCMR 4: https://www.epa.gov/dwucmr/fourth-unregulated-contaminant-monitoring-rule
- EPA UCMR occurrence data: https://www.epa.gov/dwucmr/occurrence-data-unregulated-contaminant-monitoring-rule

Downloaded EPA files are raw source materials and must not be passed directly to these scripts. They must first be transformed into the documented integrated-v1 contract outside this repository.

## UK source

The cleaned UK tables were reconstructed from publicly accessible government-sponsored reports. The reports, rather than the derived CSV files, are the authoritative public sources:

- `uk_dwi242_locked_external.csv`: Simon A. Parsons and Emma H. Goslan, *Evaluation of Haloacetic Acids Concentrations in Treated Drinking Waters*, Defra Drinking Water Inspectorate Project WT1236, Cranfield University, final report, 2011. Official DWI-hosted PDF: https://dwi-production-files.s3.eu-west-2.amazonaws.com/wp-content/uploads/2020/10/27111051/DWI70_2_242.pdf (accessed 2026-07-29).
- `uk_dbp2009_field_external.csv` and `uk_dbp2009_formation_potential.csv`: Simon A. Parsons, Emma H. Goslan, Sophie A. Rocks, Philip Holmes, Leonard S. Levy, and Stuart Krasner, *Study into the Formation of Disinfection By-products of Chloramination, Potential Health Implications and Techniques for Minimisation*, final report, 2009. Official Drinking Water Quality Regulator for Scotland-hosted PDF: https://dwqr.scot/media/fmjfelxu/research-prev-study-into-the-formation-of-disinfection-by-products-of-chloramination-may-2009.pdf (accessed 2026-07-29).
- `DWI70_2_194.pdf`, *The Formation and Occurrence of Haloacetic Acids in Drinking Water* (N. J. D. Graham, C. D. Collins, M. Nieuwenhuijsen, and M. R. Templeton; final report, June 2009), supplied provenance for records that were not reconstructable for the modeled UK cohorts. Official DWI-hosted PDF: https://dwi-production-files.s3.eu-west-2.amazonaws.com/wp-content/uploads/2020/10/27110905/DWI70_2_194.pdf (accessed 2026-07-29).

The WT1236 and 2009 chloramination reports state that their contents are copyrighted and may not be reproduced without prior permission. This code repository therefore distributes neither the reports nor the extracted UK tables. Users must obtain the reports from the official links and independently confirm any permission required to reconstruct or redistribute derived data. The manuscript must describe the UK analysis as a retrospective stress test because those outcomes had already been viewed before the locked optimization.

## Sharing checklist

Before sharing an independently prepared integrated-v1 package, verify source licenses, remove direct personal or facility-identifying fields not needed by the model, document transformations and units, record cryptographic hashes, and keep raw downloads outside the experiment repository.
