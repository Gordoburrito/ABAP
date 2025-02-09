import pytest
import pandas as pd
from pathlib import Path
from src.format_product_data_to_ABAP import format_product_data_to_ABAP


@pytest.fixture(scope="module")
def product_data():
    """Fixture to load the sample product CSV."""
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data/samples/product_data_samp.csv"
    return pd.read_csv(data_path)


@pytest.fixture(scope="module")
def transformed_data(product_data):
    """Fixture to transform the product data once per module."""
    return format_product_data_to_ABAP(product_data)


def test_transformed_data_structure(transformed_data):
    """Ensure the transformed data contains all expected columns."""
    expected_columns = {
        "Title",
        "Tag",
        "MPN",
        "Cost",
        "Price",
        "Dropship",
        "Body HTML",
        "Collection",
        "Product Type",
        "Meta Title",
        "Meta Description",
        "Notes",
    }
    actual_columns = set(transformed_data.columns)
    missing = expected_columns - actual_columns
    assert not missing, f"Missing expected columns: {missing}"


@pytest.mark.parametrize(
    "title_pattern, expected_ids",
    [
        (
            "64/65 Plymouth B-Body Automatic Shifter Slide Insert ABS Plastic",
            [
                "1964_Plymouth_Belvedere",
                "1965_Plymouth_Satellite",
                "1965_Plymouth_Belvedere",
            ],
        ),
        (
            "66/70 Satellite & Belvedere 4-Speed Cover Molded ABS Plastic",
            [
                "1966_Plymouth_Satellite",
                "1966_Plymouth_Belvedere",
                "1967_Plymouth_Satellite",
                "1967_Plymouth_Belvedere",
                "1968_Plymouth_Satellite",
                "1968_Plymouth_Belvedere",
                "1969_Plymouth_Satellite",
                "1969_Plymouth_Belvedere",
                "1970_Plymouth_Satellite",
                "1970_Plymouth_Belvedere",
            ],
        ),
        (
            "67/69 Valiant Automatic Console Kit DiNOC Gunstock Vinyl 2pc",
            ["1967_Plymouth_Valiant", "1968_Plymouth_Valiant", "1969_Plymouth_Valiant"],
        ),
        (
            "39 Plymouth 4 Door Sedan Rear ¼ Trim Tan Doorboard 2pc",
            ["1939_Plymouth_P8 Deluxe", "1939_Plymouth_P7 Roadking"],
        ),
    ],
)
def test_tag_transformations(transformed_data, title_pattern, expected_ids):
    """Verify that the Tag transformation matches the expected IDs for each product."""
    mask = transformed_data["Title"].str.contains(title_pattern, na=False)
    assert mask.any(), f"No row found with title containing '{title_pattern}'"
    row = transformed_data[mask].iloc[0]
    tag_ids = [tag.strip() for tag in row["Tag"].split(",")]
    assert sorted(tag_ids) == sorted(
        expected_ids
    ), f"Tag IDs {tag_ids} do not match expected for product with title '{title_pattern}'"
