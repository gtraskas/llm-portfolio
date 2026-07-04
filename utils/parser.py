"""Row-level parsing and curation rules for the wine-reviews dataset.

Mirrors pricer/parser.py: one function that turns a raw datapoint into a
Wine, or None if it fails any curation rule.
"""

from utils.items import Wine

# Curation constants — tune after EDA, document any change in the README
MIN_POINTS: float = 80.0
MAX_POINTS: float = 100.0
MIN_PRICE: float = 4.0  # below this is a data error in winemag data
MAX_PRICE: float = 250.0  # 91.6% of wines fall in $4-250; caps collector outliers
MIN_DESC_CHARS: int = 100  # too short -> no signal in the tasting note
MAX_DESC_CHARS: int = 2000  # signal plateaus; guards token cost


def parse(datapoint: dict) -> Wine | None:
    """Create a Wine from a raw row, or None if it fails curation.

    Rules: points and price present and in bounds, description length in
    bounds, and all context fields non-empty after stripping.

    Args:
        datapoint: One raw row from the HuggingFace dataset.

    Returns:
        A curated Wine, or None if the row should be excluded.
    """
    try:
        points = float(datapoint["points"])
        price = float(datapoint["price"])
        description = str(datapoint["description"]).strip()
        title = str(datapoint["title"] or "").strip()
        country = str(datapoint["country"] or "").strip()
        province = str(datapoint["province"] or "").strip()
        region = str(datapoint["region_1"] or "").strip()
        variety = str(datapoint["variety"] or "").strip()
        winery = str(datapoint["winery"] or "").strip()
    except (KeyError, TypeError, ValueError):
        return None

    if not (MIN_POINTS <= points <= MAX_POINTS):
        return None
    if not (MIN_PRICE <= price <= MAX_PRICE):
        return None
    if not (MIN_DESC_CHARS <= len(description) <= MAX_DESC_CHARS):
        return None
    if not (title and country and variety and winery):
        return None

    return Wine(
        title=title,
        points=points,
        price=price,
        country=country,
        province=province,
        region=region or "Unknown",
        variety=variety,
        winery=winery,
        full=description,
    )
