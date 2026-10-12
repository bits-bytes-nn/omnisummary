from unittest.mock import MagicMock, patch

import pytest

from lambda_handlers import threads_refresh_handler as h


class TestThreadsRefreshHandler:
    def test_no_token_is_noop(self):
        with patch.object(h, "resolve_secret", return_value=""):
            result = h.handler({}, None)
        assert result["statusCode"] == 200
        assert result["body"] == "no token"

    def test_resolves_the_token_strictly(self):
        # An SSM outage used to come back as "" and be reported as "no token configured, nothing to
        # refresh" — a 200 that quietly let the live token expire. strict=True makes it raise.
        with patch.object(h, "resolve_secret", return_value="OLD") as rs:
            with patch.object(h.httpx, "get") as get:
                get.return_value.is_error = False
                get.return_value.json.return_value = {"access_token": "NEW"}
                with patch.object(h.boto3, "client"):
                    h.handler({}, None)
        assert rs.call_args.kwargs == {"strict": True}

    def test_refreshes_and_writes_back_to_ssm(self, monkeypatch):
        monkeypatch.setenv("PROJECT_NAME", "omnisummary")
        monkeypatch.setenv("STAGE", "dev")
        resp = MagicMock()
        resp.is_error = False
        resp.json.return_value = {"access_token": "NEW_TOKEN", "expires_in": 5184000}
        ssm = MagicMock()
        with patch.object(h, "resolve_secret", return_value="OLD_TOKEN"):
            with patch.object(h.httpx, "get", return_value=resp) as get:
                with patch.object(h.boto3, "client", return_value=ssm):
                    result = h.handler({}, None)
        assert result["statusCode"] == 200 and result["body"] == "refreshed"
        # called the refresh endpoint with the old token
        assert get.call_args.kwargs["params"]["access_token"] == "OLD_TOKEN"
        # wrote the renewed token back, overwriting in place
        put = ssm.put_parameter.call_args.kwargs
        assert put["Name"] == "/omnisummary/dev/threads-access-token"
        assert put["Value"] == "NEW_TOKEN"
        assert put["Overwrite"] is True
        # No Type: the CFN-created parameter is a String (AWS::SSM::Parameter can't make a
        # SecureString), and sending Type=SecureString on overwrite is a rejected type change.
        assert "Type" not in put

    def test_refresh_http_failure_reraises(self):
        # A silent 500 body left the token un-refreshed with no alarm until Threads delivery
        # broke 60 days later; the failure must reach Lambda's Errors metric.
        with patch.object(h, "resolve_secret", return_value="OLD"):
            with patch.object(h.httpx, "get", side_effect=h.httpx.ConnectError("network")):
                with patch.object(h, "logger") as log:
                    with pytest.raises(RuntimeError, match="request failed"):
                        h.handler({}, None)
        assert log.error.called

    def test_a_failed_refresh_never_leaks_the_token(self):
        # httpx's error text carries the request URL, and the token is in its query string: the old
        # handler logged str(e) and re-raised it, writing the long-lived token into CloudWatch.
        secret = "THAAsecretTOKEN123"

        def _get(url, **kwargs):
            transport = h.httpx.MockTransport(lambda request: h.httpx.Response(400, text="expired"))
            with h.httpx.Client(transport=transport) as client:
                return client.get(url, **kwargs)

        with patch.object(h, "resolve_secret", return_value=secret):
            with patch.object(h.httpx, "get", side_effect=_get):
                with patch.object(h, "logger") as log:
                    with pytest.raises(RuntimeError) as raised:
                        h.handler({}, None)
        assert secret not in str(raised.value)
        assert all(secret not in str(c) for c in log.error.call_args_list)

    def test_ssm_write_failure_reraises(self):
        resp = MagicMock()
        resp.is_error = False
        resp.json.return_value = {"access_token": "NEW"}
        ssm = MagicMock()
        ssm.put_parameter.side_effect = RuntimeError("denied")
        with patch.object(h, "resolve_secret", return_value="OLD"):
            with patch.object(h.httpx, "get", return_value=resp):
                with patch.object(h.boto3, "client", return_value=ssm):
                    with pytest.raises(RuntimeError, match="denied"):
                        h.handler({}, None)
