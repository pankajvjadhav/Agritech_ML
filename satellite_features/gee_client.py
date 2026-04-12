import os

import ee


_INITIALIZED = False


def initialize_earth_engine():
    global _INITIALIZED

    if _INITIALIZED:
        return

    project_id = os.getenv("EE_PROJECT_ID", "soil-model-490412")
    service_account_email = os.getenv("EE_SERVICE_ACCOUNT_EMAIL")
    private_key_file = os.getenv("EE_PRIVATE_KEY_FILE")

    try:
        if service_account_email and private_key_file:
            credentials = ee.ServiceAccountCredentials(
                service_account_email,
                key_file=private_key_file,
            )
            ee.Initialize(credentials, project=project_id)
        else:
            ee.Initialize(project=project_id)

        _INITIALIZED = True
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine initialization failed. Set EE_PROJECT_ID and either "
            "EE_SERVICE_ACCOUNT_EMAIL + EE_PRIVATE_KEY_FILE, or configure local EE auth."
        ) from exc