"""
test_main.py

Tests para main.py.
Cubre: setup_logging (INFO y DEBUG), load_test_queries (éxito y archivo faltante),
run_single_query, run_test_suite (con y sin overall score), run_interactive
(query normal, query vacía, salir/exit/quit, KeyboardInterrupt, EOFError)
y main() en todos sus modos (demo, --test, --interactive, query arg, --no-eval, --debug).
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


class TestSetupLogging:
    def test_info_level_by_default(self):
        from main import setup_logging

        with patch("main.logging.basicConfig") as mock_basic:
            setup_logging(debug=False)
        call_kwargs = mock_basic.call_args[1]
        assert call_kwargs["level"] == logging.INFO

    def test_debug_level_when_flag_set(self):
        from main import setup_logging

        with patch("main.logging.basicConfig") as mock_basic:
            setup_logging(debug=True)
        call_kwargs = mock_basic.call_args[1]
        assert call_kwargs["level"] == logging.DEBUG

    def test_format_includes_timestamp_and_level(self):
        from main import setup_logging

        with patch("main.logging.basicConfig") as mock_basic:
            setup_logging(debug=False)
        fmt = mock_basic.call_args[1]["format"]
        assert "%(asctime)s" in fmt
        assert "%(levelname)s" in fmt

    def test_external_loggers_silenced_in_info_mode(self):
        from main import setup_logging

        with patch("main.logging.basicConfig"):
            with patch("main.logging.getLogger") as mock_get_logger:
                setup_logging(debug=False)
        # Debe haberse llamado getLogger para silenciar libs externas
        called_names = [c[0][0] for c in mock_get_logger.call_args_list if c[0]]
        assert "httpx" in called_names or "openai" in called_names

    def test_external_loggers_not_silenced_in_debug_mode(self):
        from main import setup_logging

        with patch("main.logging.basicConfig"):
            with patch("main.logging.getLogger") as mock_get_logger:
                setup_logging(debug=True)
        # En modo debug no se silencian libs externas
        called_names = [c[0][0] for c in mock_get_logger.call_args_list if c[0]]
        assert "httpx" not in called_names


# ---------------------------------------------------------------------------
# load_test_queries
# ---------------------------------------------------------------------------


class TestLoadTestQueries:
    def test_loads_queries_from_json(self, test_queries_file):
        from main import load_test_queries

        queries = load_test_queries(test_queries_file)
        assert len(queries) == 2
        assert queries[0]["query"] == "¿Cuántos días de vacaciones tengo?"

    def test_returns_list_of_dicts(self, test_queries_file):
        from main import load_test_queries

        queries = load_test_queries(test_queries_file)
        assert isinstance(queries, list)
        assert all(isinstance(q, dict) for q in queries)

    def test_missing_file_calls_sys_exit(self):
        from main import load_test_queries

        with pytest.raises(SystemExit) as exc_info:
            load_test_queries("non_existent_file_xyz.json")
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# run_single_query
# ---------------------------------------------------------------------------


class TestRunSingleQuery:
    def test_calls_system_process(self):
        from main import run_single_query

        mock_system = MagicMock()
        run_single_query(mock_system, "¿Cuántos días de vacaciones tengo?")
        mock_system.process.assert_called_once_with("¿Cuántos días de vacaciones tengo?")


# ---------------------------------------------------------------------------
# run_test_suite
# ---------------------------------------------------------------------------


class TestRunTestSuite:
    def _make_result(self, expected="hr", obtained="hr", overall=8):
        return {
            "expected_agent": expected,
            "domain": obtained,
            "routing_correct": expected == obtained,
            "evaluation": {"overall": overall} if overall is not None else None,
        }

    def test_calls_run_test_queries(self, test_queries_file):
        from main import run_test_suite

        mock_system = MagicMock()
        mock_system.run_test_queries.return_value = [self._make_result()]

        with patch("main.load_test_queries", return_value=[{"query": "q", "expected_agent": "hr"}]):
            run_test_suite(mock_system)

        mock_system.run_test_queries.assert_called_once()

    def test_prints_summary_table(self, capsys):
        from main import run_test_suite

        mock_system = MagicMock()
        mock_system.run_test_queries.return_value = [self._make_result("hr", "hr", 9)]

        with patch("main.load_test_queries", return_value=[{"query": "q", "expected_agent": "hr"}]):
            run_test_suite(mock_system)

        captured = capsys.readouterr()
        assert "hr" in captured.out

    def test_shows_na_when_no_evaluation(self, capsys):
        from main import run_test_suite

        mock_system = MagicMock()
        mock_system.run_test_queries.return_value = [self._make_result("tech", "tech", None)]

        with patch(
            "main.load_test_queries", return_value=[{"query": "q", "expected_agent": "tech"}]
        ):
            run_test_suite(mock_system)

        captured = capsys.readouterr()
        assert "N/A" in captured.out


# ---------------------------------------------------------------------------
# run_interactive
# ---------------------------------------------------------------------------


class TestRunInteractive:
    def test_processes_single_query_then_exits(self, capsys):
        from main import run_interactive

        mock_system = MagicMock()

        with patch("main.input", side_effect=["¿Cuántos días de vacaciones tengo?", "salir"]):
            run_interactive(mock_system)

        mock_system.process.assert_called_once_with("¿Cuántos días de vacaciones tengo?")

    def test_exits_on_salir(self):
        from main import run_interactive

        mock_system = MagicMock()

        with patch("main.input", side_effect=["salir"]):
            run_interactive(mock_system)

        mock_system.process.assert_not_called()

    def test_exits_on_exit(self):
        from main import run_interactive

        mock_system = MagicMock()

        with patch("main.input", side_effect=["exit"]):
            run_interactive(mock_system)

        mock_system.process.assert_not_called()

    def test_exits_on_quit(self):
        from main import run_interactive

        mock_system = MagicMock()

        with patch("main.input", side_effect=["quit"]):
            run_interactive(mock_system)

        mock_system.process.assert_not_called()

    def test_keyboard_interrupt_exits_gracefully(self, capsys):
        from main import run_interactive

        mock_system = MagicMock()

        with patch("main.input", side_effect=KeyboardInterrupt):
            run_interactive(mock_system)

        captured = capsys.readouterr()
        assert "Saliendo" in captured.out

    def test_eof_error_exits_gracefully(self, capsys):
        from main import run_interactive

        mock_system = MagicMock()

        with patch("main.input", side_effect=EOFError):
            run_interactive(mock_system)

        captured = capsys.readouterr()
        assert "Saliendo" in captured.out

    def test_empty_query_is_skipped(self):
        from main import run_interactive

        mock_system = MagicMock()

        with patch("main.input", side_effect=["", "  ", "salir"]):
            run_interactive(mock_system)

        mock_system.process.assert_not_called()

    def test_multiple_queries_before_exit(self):
        from main import run_interactive

        mock_system = MagicMock()

        with patch("main.input", side_effect=["query 1", "query 2", "salir"]):
            run_interactive(mock_system)

        assert mock_system.process.call_count == 2


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


class TestMain:
    def _mock_system_cls(self):
        mock_system = MagicMock()
        mock_system.process.return_value = {
            "domain": "hr",
            "answer": "Respuesta.",
            "evaluation": None,
            "langfuse_url": "https://cloud.langfuse.com/trace/abc",
        }
        return mock_system

    def test_demo_mode_processes_four_queries(self):
        from main import main

        mock_system = self._mock_system_cls()

        with (
            patch("main.MultiAgentSystem", return_value=mock_system),
            patch("main.setup_logging"),
            patch("sys.argv", ["main.py"]),
        ):
            main()

        assert mock_system.process.call_count == 4

    def test_test_flag_calls_run_test_suite(self):
        from main import main

        mock_system = self._mock_system_cls()

        with (
            patch("main.MultiAgentSystem", return_value=mock_system),
            patch("main.setup_logging"),
            patch("main.run_test_suite") as mock_suite,
            patch("sys.argv", ["main.py", "--test"]),
        ):
            main()

        mock_suite.assert_called_once_with(mock_system)

    def test_interactive_flag_calls_run_interactive(self):
        from main import main

        mock_system = self._mock_system_cls()

        with (
            patch("main.MultiAgentSystem", return_value=mock_system),
            patch("main.setup_logging"),
            patch("main.run_interactive") as mock_inter,
            patch("sys.argv", ["main.py", "--interactive"]),
        ):
            main()

        mock_inter.assert_called_once_with(mock_system)

    def test_query_argument_calls_run_single_query(self):
        from main import main

        mock_system = self._mock_system_cls()

        with (
            patch("main.MultiAgentSystem", return_value=mock_system),
            patch("main.setup_logging"),
            patch("main.run_single_query") as mock_single,
            patch("sys.argv", ["main.py", "¿Cuántos días de vacaciones?"]),
        ):
            main()

        mock_single.assert_called_once_with(mock_system, "¿Cuántos días de vacaciones?")

    def test_no_eval_flag_disables_evaluation(self):
        from main import main

        with (
            patch("main.MultiAgentSystem") as mock_cls,
            patch("main.setup_logging"),
            patch("sys.argv", ["main.py", "--no-eval"]),
        ):
            mock_cls.return_value = self._mock_system_cls()
            main()

        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["enable_evaluation"] is False

    def test_without_no_eval_flag_enables_evaluation(self):
        from main import main

        with (
            patch("main.MultiAgentSystem") as mock_cls,
            patch("main.setup_logging"),
            patch("sys.argv", ["main.py"]),
        ):
            mock_cls.return_value = self._mock_system_cls()
            main()

        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["enable_evaluation"] is True

    def test_debug_flag_calls_setup_logging_with_debug_true(self):
        from main import main

        with (
            patch("main.MultiAgentSystem") as mock_cls,
            patch("main.setup_logging") as mock_setup,
            patch("sys.argv", ["main.py", "--debug"]),
        ):
            mock_cls.return_value = self._mock_system_cls()
            main()

        mock_setup.assert_called_once_with(debug=True)

    def test_without_debug_flag_calls_setup_logging_with_debug_false(self):
        from main import main

        with (
            patch("main.MultiAgentSystem") as mock_cls,
            patch("main.setup_logging") as mock_setup,
            patch("sys.argv", ["main.py"]),
        ):
            mock_cls.return_value = self._mock_system_cls()
            main()

        mock_setup.assert_called_once_with(debug=False)
