# Data and Code Repository

**Manuscript:** Evaluating Regional Innovation Capability in China: Capability Tiers, Spatial Patterns, Efficiency, and Constraints

**Corresponding author:** Kai Tian, Shenzhen University of Information Technology, Shenzhen, Guangdong 518172, China
&nbsp;&nbsp;·&nbsp;&nbsp;ORCID: https://orcid.org/0000-0002-8884-9759

**Data and code repository:** https://github.com/KaiTian4756/regional-innovation-capability-china
**License:** CC BY 4.0 (see Section 6)

---

## 1. Overview

This repository contains the panel dataset, analytical result files, and Python scripts
underlying the manuscript. The study covers 31 Chinese provincial-level regions over
2011–2020 (310 province-year observations) and evaluates regional innovation capability
using an entropy-weighted TOPSIS composite index (the innovation capability index) built
from 20 secondary indicators grouped into five dimensions:

1. Scientific research talent capability (D1)
2. Technological investment capability (D2)
3. High-tech output capability (D3)
4. Technology-oriented enterprise capability (D4)
5. Innovation environment capability (D5)

Subsequent analyses reported in the manuscript are: K-means capability tiers (2020, k = 4),
global Moran's I spatial dependence, input-oriented DEA-BCC efficiency, two-way
fixed-effects panel regression, sigma/beta convergence, and obstacle degree analysis.

All terminology in these files follows the manuscript (Tables 1–5) exactly.

## 2. File inventory

### Data

| File | Description |
|---|---|
| `panel_with_scores.csv` | Full panel (31 provinces × 2011–2020). Columns: `Province`, `Year`, `Innovation capability index` (entropy-TOPSIS composite score), the five dimension scores, and the 20 raw secondary indicators X1–X20 (see Section 3). |

### Result files

| File | Description | Manuscript reference |
|---|---|---|
| `clustering_results.csv` | K-means clustering results (2020, k = 4): cluster IDs and capability tiers (Tier 1 High / Tier 2 Upper-middle / Tier 3 Lower-middle / Tier 4 Low). | Tables 2–3, Figure 3 |
| `dea_results.csv` | Input-oriented DEA-BCC efficiency scores for 2020 (with efficiency ranks) and annual mean efficiency 2011–2020. | Section 3.4, Figure 5 |
| `panel_regression_results.csv` | Panel regression results with province-clustered standard errors for four specifications; the province + year fixed-effects four-predictor model is the one reported in Table 4. | Table 4 |
| `convergence_analysis.csv` | Sigma convergence (annual CV, 2011–2020), national absolute beta convergence, and four-region club beta convergence. | Table 5, Figure 6 |
| `obstacle_analysis.csv` | Obstacle degree shares (%) of the five dimensions by province, 2020, with each province's most constraining dimension. | Section 3.5, Figure 6 |
| `spatial_analysis_results.csv` | Global Moran's I, Z value, p value, and significance by year, 2011–2020. | Section 3.3, Figure 4 |

### Code (`code/`)

| Script | Purpose |
|---|---|
| `code/run_diagnostics_and_fe.py` | Five-dimension structure diagnostics (correlation, PCA, leave-one-province-out influence) and the four fixed-effects/OLS panel regressions, including the recommended province + year FE specification reported in Table 4. |
| `code/run_dea_convergence_consistency.py` | Recomputes input-oriented DEA-BCC efficiency (BCC envelopment LP with convexity constraint sum λ = 1), sigma/beta/club convergence, and cross-file numerical consistency checks. |
| `code/run_kmeans_verification.py` | Reproduces the reported k = 4 tiers from the one-dimensional index and evaluates standardized five-dimension K-means for k = 2–6 (silhouette, Calinski–Harabasz, Davies–Bouldin; initialization and leave-one-province-out ARI stability). |

All scripts read the CSV files in this repository and write outputs to
`./verification_output/` (created at run time; not distributed).

### Repository files

| File | Description |
|---|---|
| `README.md` | This file: file inventory, variable definitions, reproduction instructions. |
| `LICENSE` | CC BY 4.0 license text and the required attribution string. |

In total the deposit contains 1 panel dataset, 6 result files, 3 Python scripts, and 2
documentation/license files.

## 3. Variable definitions

### 3.1 Identifier and score columns (`panel_with_scores.csv`)

- `Province` — provincial-level region name (31 regions).
- `Year` — 2011–2020.
- `Innovation capability index` — entropy-weighted TOPSIS composite score (0–100 scale), referred to in the manuscript as the regional innovation capability index (RICI).
- Five dimension scores, named exactly as in the manuscript: `Scientific research talent capability`, `Technological investment capability`, `High-tech output capability`, `Technology-oriented enterprise capability`, `Innovation environment capability`. Each is the entropy-weighted aggregate of its secondary indicators.

### 3.2 Secondary indicators X1–X20 (wording follows Table 1 of the manuscript)

| Code | Indicator | Dimension |
|---|---|---|
| X1 | Researchers per 10,000 population | D1 |
| X2 | Full-time university teachers | D1 |
| X3 | Students enrolled in higher education per 10,000 population | D1 |
| X4 | Employees in high-tech zones as a proportion of population | D1 |
| X5 | Returned overseas researchers in high-tech zones as a proportion of population | D1 |
| X6 | R&D expenditure as a proportion of GDP | D2 |
| X7 | Government fiscal expenditure on science and technology as a proportion of total fiscal expenditure | D2 |
| X8 | Government fiscal expenditure on education as a proportion of total fiscal expenditure | D2 |
| X9 | Internal R&D expenditure of high-tech zone enterprises as a proportion of GDP | D2 |
| X10 | Industrial output value of high-tech zone enterprises as a proportion of GDP | D3 |
| X11 | Newly granted patents | D3 |
| X12 | Technology contract transaction value as a proportion of regional GDP | D3 |
| X13 | Export value of high-tech zone industries as a proportion of total commodity exports | D3 |
| X14 | Number of high-tech enterprises | D4 |
| X15 | Number of industrial enterprises above designated size | D4 |
| X16 | Number of national key laboratories | D5 |
| X17 | Number of universities | D5 |
| X18 | Number of research institutes | D5 |
| X19 | Number of national science and technology enterprise incubators | D5 |
| X20 | Number of maker spaces | D5 |

All indicators are positive-direction ("+"), as specified in Table 1 of the manuscript.
Data sources: China Statistical Yearbook, China High-Tech Industry Statistical Yearbook,
China Torch Statistical Yearbook, and provincial statistical yearbooks (2012–2021 editions),
as described in Section 2.2 of the manuscript.

### 3.3 Regional grouping (used in `convergence_analysis.csv` and Table 5)

- **Eastern** (10): Beijing, Tianjin, Hebei, Shanghai, Jiangsu, Zhejiang, Fujian, Shandong, Guangdong, Hainan
- **Central** (6): Shanxi, Anhui, Jiangxi, Henan, Hubei, Hunan
- **Western** (12): Inner Mongolia, Guangxi, Chongqing, Sichuan, Guizhou, Yunnan, Tibet, Shaanxi, Gansu, Qinghai, Ningxia, Xinjiang
- **Northeastern** (3): Liaoning, Jilin, Heilongjiang

## 4. Reproducing the results

Requirements: Python 3.10+ with `numpy`, `pandas`, and `scipy`.

```bash
python code/run_diagnostics_and_fe.py
python code/run_dea_convergence_consistency.py
python code/run_kmeans_verification.py
```

Each script creates `./verification_output/` and writes recomputed results plus
side-by-side comparisons against the distributed result files. The scripts do not modify
any distributed data file.

Expected key values (as reported in the manuscript):

- 2020 mean DEA-BCC efficiency: 0.7847; 10 provinces on the efficiency frontier.
- Recommended two-way FE model (Table 4): Scientific research talent 0.3319 (clustered SE 0.0865), Technological investment 0.0294 (0.0787), Technology-oriented enterprises 0.2287 (0.0465), Innovation environment 0.1260 (0.0754); within R² = 0.6057; n = 310.
- National absolute beta convergence 2011–2020: β = 0.0506, SE = 0.0687, R² = 0.0184, n = 31.

## 5. Notes and limitations

- **Spatial analysis.** The spatial weights matrix and province geometry used for the
  Moran's I calculations are not distributed in this repository; Moran's I was therefore
  not independently recomputed by the verification scripts. The reported values are
  distributed in `spatial_analysis_results.csv` for cross-checking.
- **DEA interpretation.** DEA scores measure innovation resource-conversion efficiency
  conditional on the observed input scale; they are not measures of overall innovation
  capability or output leadership.
- **Fixed-effects estimates.** The regression coefficients are association estimates in a
  31-province, 2011–2020 panel; they do not establish causality.
- **K-means tiers.** The capability tiers are a static 2020 cross-sectional grouping;
  they are not evidence of dynamic catching-up processes.

## 6. License

The data and code in this deposit are released under the Creative Commons Attribution 4.0
International License (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/). You are free
to share and adapt the material for any purpose, provided that appropriate credit is given,
a link to the license is included, and any changes are indicated.

## 7. Citation

If you use these data or code, please cite the manuscript and the archived dataset:

```
Tian K. Evaluating Regional Innovation Capability in China: Capability Tiers, Spatial
Patterns, Efficiency, and Constraints. Humanities and Social Sciences Communications
(under review).

```


