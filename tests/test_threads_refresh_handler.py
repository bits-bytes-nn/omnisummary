from unittest.mock import MagicMock, patch

from lambda_handlers import threads_refresh_handler as h


class TestThreadsRefreshHandler:
    def test_no_token_is_noop(self):
        with patch.object(h, "resolve_secret", return_value=""):
            result = h.handler({}, None)
        assert result["statusCode"] == 200
        assert result["body"] == "no token"

    def test_refreshes_and_writes_back_to_ssm(self, monkeypatch):
        monkeypatch.setenv("PROJECT_NAME", "omnisummary")
        monkeypatch.setenv("STAGE", "dev")
        resp = MagicMock()
        resp.json.return_value = {"access_token": "NEW_TOKEN", "expires_in": 5184000}
        ssm = MagicMock()
        with patch.object(h, "resolve_secret", return_value="OLD_TOKEN"):
            with patch.object(h.httpx, "get", return_value=resp) as get:
                with patch.object(h.boto3, "client", return_value=ssm):
                    result = h.handler({}, None)
        assert result["statusCode"] == 200 and result["body"] == "refreshed"
        # called the refresh endpoint with the old token
        assert get.call_args.kwargs["params"]["access_token"] == "OLD_TOKEN"
        # wrote the renewed token back as a SecureString
        put = ssm.put_parameter.call_args.kwargs
        assert put["Name"] == "/omnisummary/dev/threads-access-token"
        assert put["Value"] == "NEW_TOKEN"
        assert put["Type"] == "SecureString" and put["Overwrite"] is True

    def test_refresh_http_failure_returns_500(self):
        with patch.object(h, "resolve_secret", return_value="OLD"):
            with patch.object(h.httpx, "get", side_effect=RuntimeError("network")):
                result = h.handler({}, None)
        assert result["statusCode"] == 500 and result["body"] == "refresh failed"
