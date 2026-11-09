"""Association-rule-based service recommendations using the Armut case dataset."""

from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

DATA_PATH = Path(__file__).resolve().parent / "armut_data.csv"


def load_and_prepare_data(path: Path) -> pd.DataFrame:
    """Read purchases and assign each user-month a basket identifier."""
    data = pd.read_csv(path)
    required = {"UserId", "ServiceId", "CategoryId", "CreateDate"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    data["CreateDate"] = pd.to_datetime(data["CreateDate"])
    data["Service"] = data["ServiceId"].astype(str) + "_" + data["CategoryId"].astype(str)
    data["YearMonth"] = data["CreateDate"].dt.strftime("%Y-%m")
    data["BasketID"] = data["UserId"].astype(str) + "_" + data["YearMonth"]
    return data


def build_basket_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Return a boolean basket-by-service presence matrix."""
    return pd.crosstab(data["BasketID"], data["Service"]).gt(0)


def generate_rules(baskets: pd.DataFrame, min_support: float = 0.01) -> pd.DataFrame:
    """Discover frequent service sets and their association rules."""
    itemsets = apriori(baskets, min_support=min_support, use_colnames=True)
    return association_rules(itemsets, metric="support", min_threshold=min_support)


def arl_recommender(rules_df: pd.DataFrame, service_id: str, rec_count: int = 3) -> list[str]:
    """Recommend distinct consequent services, prioritizing highest-lift rules."""
    recommendations = []
    for _, rule in rules_df.sort_values("lift", ascending=False).iterrows():
        if service_id in rule["antecedents"]:
            for suggested in sorted(rule["consequents"]):
                if suggested != service_id and suggested not in recommendations:
                    recommendations.append(suggested)
        if len(recommendations) >= rec_count:
            break
    return recommendations[:rec_count]


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Place an authorized copy of armut_data.csv next to this script."
        )
    data = load_and_prepare_data(DATA_PATH)
    baskets = build_basket_matrix(data)
    rules = generate_rules(baskets)
    recommendations = arl_recommender(rules, "2_0", 3)
    print(f"Transactions: {len(data):,}")
    print(f"Baskets: {baskets.shape[0]:,} | Services: {baskets.shape[1]}")
    print(f"Association rules: {len(rules)}")
    print(f"Recommendations for service 2_0: {recommendations}")


if __name__ == "__main__":
    main()
