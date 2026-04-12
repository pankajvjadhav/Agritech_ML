# app/nutrient_value_mapper.py

STATUS_TO_VALUE = {
    "N": {
        "Low": 210,
        "Medium": 420,
        "High": 620
    },
    "P": {
        "Low": 6,
        "Medium": 18,
        "High": 30
    },
    "K": {
        "Low": 90,
        "Medium": 200,
        "High": 320
    },

    "OC": {
        "Low": 0.4,
        "Medium": 0.6,
        "High": 0.9
    },

    "EC": {
        "Non-Saline": 0.6,
        "Saline": 2.5
    },

    "S": {
        "Low": 8,
        "Medium": 15,
        "High": 25
    },
    "Fe": {
        "Deficient": 3.5,
        "Sufficient": 6.5,
        "Excess": 10.0
    },
    "Zn": {
        "Deficient": 0.4,
        "Sufficient": 0.9,
        "Excess": 1.4
    },
    "Cu": {
        "Deficient": 0.15,
        "Sufficient": 0.5,
        "Excess": 1.0
    },
    "Mn": {
        "Deficient": 0.8,
        "Sufficient": 3.0,
        "Excess": 6.0
    }
}

NUTRIENT_UNITS = {
    "N": "kg/ha",
    "P": "kg/ha",
    "K": "kg/ha",
    "OC": "%",
    "EC": "dS/m",
    "S": "mg/kg",
    "Fe": "mg/kg",
    "Zn": "mg/kg",
    "Cu": "mg/kg",
    "Mn": "mg/kg",
}


def map_status_to_value(nutrient: str, status: str):

    nutrient = nutrient.strip()
    status = status.strip()

    if nutrient not in STATUS_TO_VALUE:
        return {
            "value": None,
            "unit": "",
            "note": "unknown nutrient"
        }

    if status not in STATUS_TO_VALUE[nutrient]:
        return {
            "value": None,
            "unit": NUTRIENT_UNITS.get(nutrient, ""),
            "note": "unknown status"
        }

    return {
        "value": STATUS_TO_VALUE[nutrient][status],
        "unit": NUTRIENT_UNITS.get(nutrient, ""),
        "note": "icar_reference"
    }