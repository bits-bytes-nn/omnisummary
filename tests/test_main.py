from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import main


@pytest.fixture(autouse=True)
def _adapters(monkeypatch):
    monkeypatch.setattr(main, "bootstrap", MagicMock())


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
                collect = AsyncMock(return_value=([MagicMock()], MagicMock()))
                with patch.object(main, "run_collectors_with_health", new=collect):
                    with patch.object(main, "run_pipeline", new=AsyncMock(return_value=(None, None, None))):
                        assert await main.main() == 1


async def test_a_local_run_stores_the_digest_before_posting_it_inline():
    # A store that failed after the post went up would leave the day's posted digest stored nowhere.
    config = MagicMock()
    config.aws.timezone = "Asia/Seoul"
    window = (date(2026, 10, 26), datetime(2026, 10, 26, 15, tzinfo=UTC))
    order: list[str] = []
    digest = MagicMock()
    with patch("sys.argv", ["main.py", "--force-republish"]):
        with patch.object(main.Config, "load", return_value=config):
            with patch.object(main, "resolve_digest_window", return_value=window):
                collect = AsyncMock(return_value=([MagicMock()], MagicMock()))
                with patch.object(main, "run_collectors_with_health", new=collect):
                    with patch.object(main, "run_pipeline", new=AsyncMock(return_value=([], [MagicMock()], digest))):
                        with patch.object(main, "persist_digest", side_effect=lambda *a: order.append("persist")):
                            publish = AsyncMock(side_effect=lambda *a, **k: order.append("publish"))
                            with patch.object(main, "publish_inline", new=publish):
                                assert await main.main() == 0
    assert order == ["persist", "publish"]
    assert publish.await_args.kwargs["force_republish"] is True
    main.bootstrap.memory_store.assert_any_call(main.Path(main.LocalPaths.DIGEST_STATE_DIR.value))


async def test_an_incomplete_local_post_exits_nonzero():
    config = MagicMock()
    config.aws.timezone = "Asia/Seoul"
    window = (date(2026, 10, 26), datetime(2026, 10, 26, 15, tzinfo=UTC))
    failed = AsyncMock(side_effect=main.PublishIncomplete("5/6 posts"))
    with patch("sys.argv", ["main.py"]):
        with patch.object(main.Config, "load", return_value=config):
            with patch.object(main, "resolve_digest_window", return_value=window):
                with patch.object(main, "already_published", return_value=False):
                    collect = AsyncMock(return_value=([MagicMock()], MagicMock()))
                    with patch.object(main, "run_collectors_with_health", new=collect):
                        with patch.object(
                            main, "run_pipeline", new=AsyncMock(return_value=([], [MagicMock()], MagicMock()))
                        ):
                            with patch.object(main, "persist_digest"), patch.object(main, "publish_inline", new=failed):
                                assert await main.main() == 1


async def test_a_dry_run_stores_and_posts_nothing():
    config = MagicMock()
    config.aws.timezone = "Asia/Seoul"
    window = (date(2026, 10, 26), datetime(2026, 10, 26, 15, tzinfo=UTC))
    with patch("sys.argv", ["main.py", "--dry-run"]):
        with patch.object(main.Config, "load", return_value=config):
            with patch.object(main, "resolve_digest_window", return_value=window):
                collect = AsyncMock(return_value=([MagicMock()], MagicMock()))
                with patch.object(main, "run_collectors_with_health", new=collect):
                    with patch.object(
                        main, "run_pipeline", new=AsyncMock(return_value=([], [MagicMock()], MagicMock()))
                    ):
                        with patch.object(main, "persist_digest") as persist:
                            with patch.object(main, "publish_inline", new=AsyncMock()) as publish:
                                assert await main.main() == 0
                        assert main.run_pipeline.await_args.kwargs["dry_run"] is True
    persist.assert_not_called()
    publish.assert_not_awaited()
