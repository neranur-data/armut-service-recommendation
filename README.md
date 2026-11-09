# Service Recommendation System with Association Rule Learning

A Python portfolio project based on the **Armut** service marketplace case study from the **Miuul Data Science Bootcamp**. The objective is to discover which services tend to occur together in customers' monthly purchases and generate simple, interpretable service recommendations.

## Business problem

A service marketplace wants to suggest relevant additional services based on historical purchase patterns. This project applies the **Apriori** algorithm and **association rules**, ranking candidate recommendations by **lift**.

## Dataset

The provided case-study dataset contains **162,523 transactions** and four original columns:

| Column | Description |
| --- | --- |
| `UserId` | Anonymized customer identifier |
| `ServiceId` | Service identifier |
| `CategoryId` | Service category identifier |
| `CreateDate` | Purchase timestamp |

A service is identified by the combination `ServiceId_CategoryId`. Each customer's purchases within a calendar month are grouped into one basket. In the original exercise, this produces **71,220 monthly baskets** across **50 distinct services**.

**Data availability:** The original dataset is not included in this public-ready package. Only upload or distribute it if the dataset owner explicitly permits redistribution. Place an authorized local copy named `armut_data.csv` alongside the script to reproduce the analysis.

## Approach

1. Load and validate the transaction data.
2. Construct combined service identifiers and customer-month basket IDs.
3. Build a binary basket–service matrix.
4. Use Apriori with `min_support=0.01` to identify frequent service combinations.
5. Generate association rules and inspect **support**, **confidence**, and **lift**.
6. Recommend distinct consequent services from matching rules, ordered by lift.

### Interpreting the metrics

- **Support:** How often a service combination appears across all baskets.
- **Confidence:** Among baskets containing service A, how often service B also appears.
- **Lift:** How much more often A and B co-occur than expected under independence. Lift greater than 1 indicates positive association, not causation.

## Example result

For a customer whose selected service is `2_0`, the exercise produces:

```text
['22_0', '25_0', '15_1']
```

Service IDs are anonymized, so the output cannot be translated into named services without an additional mapping.

## Run locally

Python 3.10+ is recommended.

```bash
python -m pip install -r requirements.txt
python armut_recommender.py
```

Make sure an authorized copy of `armut_data.csv` is in the same folder as `armut_recommender.py`.

## Limitations and next steps

- This is an educational, rule-based recommender—not a deployed production model.
- Monthly baskets are a modeling assumption and do not represent individual checkout sessions.
- Recommendations are based on observed associations, not causal effects or individual preferences.
- Future work could evaluate recommendations on a time-based holdout set and compare with popularity-based baselines.

## Acknowledgment

Case-study concept and learning materials: **Miuul Data Science Bootcamp / Armut Association Rule Learning exercise**. Code and explanations in this repository are organized as an independent portfolio implementation.
