from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import main


@pytest.mark.parametrize(
    ("argv", "collects"),
    [([], False), (["--dry-run"], True), (["--force-republish"], True)],
)
async def test_a_date_already_on_threads_runs_only_when_asked(argv, collects):
    # A second run for a posted date used to persist its own snapshot, leads and ledger over the record.
    window = (date(2026, 10, 26), datetime(2026, 10, 26, 15, tzinfo=UTC))
    with patch("sys.argv", ["main.py", *argv]):
        config = MagicMock()
        config.aws.timezone = "Asia/Seoul"
        with patch.object(main.Config, "load", return_value=config):
            with patch.object(main, "resolve_digest_window", return_value=window):
                with patch.object(main, "already_published", return_value=True):
                    with patch.object(main, "boto3"), patch.object(main, "BedrockLanguageModelFactory"):
                        collect = AsyncMock(return_value=([], MagicMock()))
                        with patch.object(main, "run_collectors_with_health", new=collect):
                            code = await main.main()
    assert collect.await_count == (1 if collects else 0)
    assert code == (1 if collects else 0)  # collecting nothing is a failure; a skipped posted day is not


async def test_a_future_date_is_refused():
    config = MagicMock()
    config.aws.timezone = "Asia/Seoul"
    with patch("sys.argv", ["main.py", "--date", "2999-01-01"]):
        with patch.object(main.Config, "load", return_value=config):
            with pytest.raises(SystemExit):
                await main.main()


def test_an_unknown_source_is_refused():
    with patch("sys.argv", ["main.py", "--sources", "x"]):
        with pytest.raises(SystemExit):
            import asyncio

            asyncio.run(main.main())


async def test_no_digest_exits_nonzero():
    config = MagicMock()
    config.aws.timezone = "Asia/Seoul"
    window = (date(2026, 10, 26), datetime(2026, 10, 26, 15, tzinfo=UTC))
    with patch("sys.argv", ["main.py", "--dry-run"]):
        with patch.object(main.Config, "load", return_value=config):
            with patch.object(main, "resolve_digest_window", return_value=window):
                with patch.object(main, "boto3"), patch.object(main, "BedrockLanguageModelFactory"):
                    collect = AsyncMock(return_value=([MagicMock()], MagicMock()))
                    with patch.object(main, "run_collectors_with_health", new=collect):
                        with patch.object(main, "run_pipeline", new=AsyncMock(return_value=(None, None, None))):
                            assert await main.main() == 1
