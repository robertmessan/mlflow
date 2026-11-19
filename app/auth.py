import msal
import os
from dotenv import load_dotenv

load_dotenv()

def acquire_access_token() -> str:
    tenant_id = os.getenv("TENANT_ID")
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")
    scope = os.getenv("SCOPE")

    authority = f"https://login.microsoftonline.com/{tenant_id}"

    if client_secret:
        app = msal.ConfidentialClientApplication(
            client_id=client_id,
            client_credential=client_secret,
            authority=authority,
        )
        result = app.acquire_token_for_client(scopes=[scope])
    else:
        app = msal.PublicClientApplication(
            client_id=client_id,
            authority=authority,
        )
        accounts = app.get_accounts()
        result = None

        if accounts:
            result = app.acquire_token_silent(scopes=[scope], account=accounts[0])

        if not result:
            flow = app.initiate_device_flow(scopes=[scope])
            if "user_code" not in flow:
                raise RuntimeError(f"Failed to create device flow: {flow}")
            print(flow["message"])
            result = app.acquire_token_by_device_flow(flow)

    if "access_token" not in result:
        raise RuntimeError(f"Unable to acquire Azure AD token: {result}")

    return result["access_token"]