NUTRIENT_RANGES = {
    "pH": {
        "low": (0, 6.5),
        "medium": (6.5, 7.0),
        "high": (7.0, float("inf"))
    },
    "EC": {
        "normal": (0, 4.0),
        "saline": (4.0, float("inf"))
    },
    "OC": {
        "low": (0, 0.50),
        "medium": (0.50, 0.75),
        "high": (0.75, float("inf"))
    },
    "N": {
        "low": (0, 280),
        "medium": (280, 560),
        "high": (560, float("inf"))
    },
    "P": {
        "low": (0, 10),
        "medium": (10, 25),
        "high": (25, float("inf"))
    },
    "K": {
        "low": (0, 120),
        "medium": (120, 280),
        "high": (280, float("inf"))
    },
    "S": {
        "deficient": (0, 10),
        "sufficient": (10, float("inf"))
    },
    "Zn": {
        "deficient": (0, 0.6),
        "sufficient": (0.6, float("inf"))
    },
    "Cu": {
        "deficient": (0, 0.2),
        "sufficient": (0.2, float("inf"))
    },
    "Fe": {
        "deficient": (0, 4.5),
        "sufficient": (4.5, float("inf"))
    },
    "Mn": {
        "deficient": (0, 2.0),
        "sufficient": (2.0, float("inf"))
    }
}
