import requests


def model_capabilities(host: str, name: str) -> list[str] | None:
    try:
        response = requests.post(f"{host}/api/show", json={"model": name}, timeout=5)
        response.raise_for_status()
        capabilities = response.json().get("capabilities")
        if isinstance(capabilities, list):
            return [item for item in capabilities if isinstance(item, str)]
    except (requests.RequestException, ValueError, AttributeError):
        pass
    return None
