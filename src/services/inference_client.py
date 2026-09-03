"""HTTP client used by the App Service to call the detector runtime."""

from typing import Protocol, cast

import httpx

from src.models.inference_schemas import (
    InferenceBatchRequest,
    InferenceBatchResponse,
    InferenceRequest,
    InferenceResponse,
)

INFERENCE_AUTH_HEADER = "X-Label-Guardian-Inference-Token"


class InferenceClientError(RuntimeError):
    """Raised when the remote inference service cannot return detections."""


class InferenceClient(Protocol):
    async def detect(self, request: InferenceRequest) -> InferenceResponse: ...
    async def detect_batch(self, request: InferenceBatchRequest) -> InferenceBatchResponse: ...


class RemoteInferenceClient:
    def __init__(
        self,
        *,
        base_url: str,
        token: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.endpoint = f"{base_url.rstrip('/')}/v1/detect"
        self.batch_endpoint = f"{base_url.rstrip('/')}/v1/detect-batch"
        self.token = token
        self.timeout = httpx.Timeout(timeout_seconds)

    async def _post(self, endpoint: str, request: InferenceRequest | InferenceBatchRequest) -> object:
        headers = {INFERENCE_AUTH_HEADER: self.token} if self.token else None
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    endpoint,
                    headers=headers,
                    json=request.model_dump(mode="json", by_alias=True, exclude_none=True),
                )
        except httpx.TimeoutException as error:
            raise InferenceClientError("Inference service timed out.") from error
        except httpx.HTTPError as error:
            raise InferenceClientError(f"Inference service request failed: {error}") from error

        if response.status_code >= 400:
            detail = response.text
            try:
                error_payload = response.json()
                if isinstance(error_payload, dict) and error_payload.get("detail"):
                    detail = str(error_payload["detail"])
            except ValueError:
                pass
            raise InferenceClientError(
                f"Inference service returned HTTP {response.status_code}: {detail}"
            )

        try:
            return response.json()
        except ValueError as error:
            raise InferenceClientError("Inference service returned an invalid response contract.") from error

    async def detect(self, request: InferenceRequest) -> InferenceResponse:
        response_payload = await self._post(self.endpoint, request)
        return cast(InferenceResponse, InferenceResponse.model_validate(response_payload))

    async def detect_batch(self, request: InferenceBatchRequest) -> InferenceBatchResponse:
        response_payload = await self._post(self.batch_endpoint, request)
        return cast(InferenceBatchResponse, InferenceBatchResponse.model_validate(response_payload))
