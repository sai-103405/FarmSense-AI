from src.utils.validators import validate_field_data


def test_valid_field_data():
    result = validate_field_data(
        nitrogen=90,
        phosphorus=42,
        potassium=43,
        temperature=20.9,
        humidity=82,
        ph=6.5,
        rainfall=202.9
    )

    assert result is True


def test_negative_nitrogen():
    try:
        validate_field_data(
            nitrogen=-1,
            phosphorus=42,
            potassium=43,
            temperature=20.9,
            humidity=82,
            ph=6.5,
            rainfall=202.9
        )
        assert False
    except ValueError:
        assert True


def test_invalid_humidity():
    try:
        validate_field_data(
            nitrogen=90,
            phosphorus=42,
            potassium=43,
            temperature=20.9,
            humidity=120,
            ph=6.5,
            rainfall=202.9
        )
        assert False
    except ValueError:
        assert True


def test_invalid_ph():
    try:
        validate_field_data(
            nitrogen=90,
            phosphorus=42,
            potassium=43,
            temperature=20.9,
            humidity=82,
            ph=15,
            rainfall=202.9
        )
        assert False
    except ValueError:
        assert True
