from __future__ import annotations

import math

import pytest

from marlib.adapters.calc import do_calculate, safe_eval


class TestSafeEval:
    def test_arithmetic(self):
        assert safe_eval("1 + 2 * 3") == 7.0

    def test_division(self):
        assert safe_eval("10 / 4") == 2.5

    def test_power_and_modulo(self):
        assert safe_eval("2 ** 10 % 7") == 2.0

    def test_unary(self):
        assert safe_eval("-3 + +4") == 1.0

    def test_round_two_args(self):
        assert safe_eval("round(456.78 / 123, 2)") == 3.71

    def test_builtins(self):
        assert safe_eval("max(1, 2, 3) + abs(-4)") == 7.0
        assert safe_eval("min(5, 2, 9)") == 2.0

    def test_constants(self):
        assert safe_eval("pi") == pytest.approx(math.pi)
        assert safe_eval("2 * e") == pytest.approx(2 * math.e)

    def test_whitespace_tolerated(self):
        assert safe_eval("  1 + 1  ") == 2.0

    def test_blocks_import_escape(self):
        with pytest.raises(Exception):
            safe_eval('__import__("os").system("echo pwned")')

    def test_blocks_attribute_access(self):
        with pytest.raises(Exception):
            safe_eval('(1).__class__')

    def test_division_by_zero_raises(self):
        with pytest.raises(Exception):
            safe_eval("1 / 0")


class TestDoCalculate:
    def test_formats_result(self):
        assert do_calculate("2 + 2") == "2 + 2 = 4.0"

    def test_catches_errors(self):
        out = do_calculate("1 / 0")
        assert out.startswith("Error evaluating '1 / 0':")

    def test_catches_sandbox_violation(self):
        out = do_calculate('__import__("os")')
        assert out.startswith("Error evaluating")
