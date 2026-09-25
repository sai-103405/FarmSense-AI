def validate_field_data(
    nitrogen,
    phosphorus,
    potassium,
    temperature,
    humidity,
    ph,
    rainfall
):
    """
    Validate soil and weather inputs.
    """

    values = {
        "nitrogen": nitrogen,
        "phosphorus": phosphorus,
        "potassium": potassium,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }

    # Check for missing values
    for name, value in values.items():
        if value is None:
            raise ValueError(f"{name} cannot be empty")

    # Basic physical/range validation
    if nitrogen < 0:
        raise ValueError("Nitrogen cannot be negative")

    if phosphorus < 0:
        raise ValueError("Phosphorus cannot be negative")

    if potassium < 0:
        raise ValueError("Potassium cannot be negative")

    if humidity < 0 or humidity > 100:
        raise ValueError("Humidity must be between 0 and 100")

    if ph < 0 or ph > 14:
        raise ValueError("pH must be between 0 and 14")

    if rainfall < 0:
        raise ValueError("Rainfall cannot be negative")

    return True