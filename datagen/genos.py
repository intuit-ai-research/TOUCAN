import os
from typing import Any

from dotenv import load_dotenv
load_dotenv()

def _load_keys_from_env() -> dict[str, str]:
    keys = {
        "APP_SECRET": os.getenv("APP_SECRET"),
        "METADATA_EXPERIENCE_ID": os.getenv("METADATA_EXPERIENCE_ID"),
        "METADATA_ORIGINATING_ASSET_ALIAS": os.getenv("METADATA_ORIGINATING_ASSET_ALIAS"),
    }
    return keys


def initialize_intuit_auth(keys: dict[str, str]):
    # Intuit authentication setup
    os.environ["APP_SECRET"] = keys["APP_SECRET"]

    METADATA_HEADERS = "headers"
    METADATA_CORP_ID = "corp_id"
    METADATA_EMAIL = "email"
    ENV_VAR_APP_ID = "APP_ID"

    context: dict[str, Any] = {
        "METADATA_EXPERIENCE_ID": keys["METADATA_EXPERIENCE_ID"],
        "METADATA_ORIGINATING_ASSET_ALIAS": keys["METADATA_ORIGINATING_ASSET_ALIAS"],
    }

    def update_credentials(context: dict[str, Any]):
        from coreaiauth import AuthConfig, AuthnClient, Environment, AuthType  # type: ignore

        #  pip install intlgntsys-mlplatform.coreaiauth.coreaiauth==2.1.0 --upgrade --extra-index-url=https://artifact.intuit.com/artifactory/api/pypi/pypi-intuit/simple;
        os.environ[ENV_VAR_APP_ID] = context["METADATA_ORIGINATING_ASSET_ALIAS"]

        client = AuthnClient(
            AuthConfig(env=Environment.E2E, authType=AuthType.USER, appId=context["METADATA_ORIGINATING_ASSET_ALIAS"])  # type: ignore
        )
        token_details = client.generate_header()
        context.update(
            {
                METADATA_HEADERS: token_details.headers,
                METADATA_CORP_ID: token_details.corpId,
                METADATA_EMAIL: token_details.email,
            }
        )

    # Update credentials
    update_credentials(context)

    INTUIT_GENOS_HEADER = {
        "intuit_experience_id": context["METADATA_EXPERIENCE_ID"],
        "intuit_originating_assetalias": context["METADATA_ORIGINATING_ASSET_ALIAS"],
    }

    INTUIT_AUTHN_HEADERS = INTUIT_GENOS_HEADER | context[METADATA_HEADERS]

    BASE_URL_PER_ENV = {
        "QAL": "https://llmexecution-qal.api.intuit.com/v3/{intuit_genos_model_id}",
        "E2E": "https://llmexecution-e2e.api.intuit.com/v3/{intuit_genos_model_id}",
        "PRF": "https://llmexecution-prf.api.intuit.com/v3/{intuit_genos_model_id}",
        "E2E_LONG": "https://llmexecution-e2e.api.intuit.com/v3/lt/{intuit_genos_model_id}",
    }

    return INTUIT_AUTHN_HEADERS, BASE_URL_PER_ENV

INTUIT_AUTHN_HEADERS, BASE_URL_PER_ENV = initialize_intuit_auth(_load_keys_from_env())