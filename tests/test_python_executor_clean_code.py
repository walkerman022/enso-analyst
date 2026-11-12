#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for _clean_code_string parsing-boundary hardening (PR #95).

Covers the review checklist: malformed JSON, deeply nested JSON,
nested code fields, the recursion depth limit, and process-control
exception propagation.
"""

import json
import sys

import pytest

from LightAgent.builtin_tools import python_executor
from LightAgent.builtin_tools.python_executor import (
    _MAX_CLEAN_DEPTH,
    _clean_code_string,
    _parse_code_parameter,
    _try_json_loads,
)


class TestMalformedJson:

    def test_malformed_json_returns_input_unchanged(self):
        malformed = '{"code": "print(1)"'  # missing closing brace
        assert _clean_code_string(malformed) == malformed

    def test_quoted_but_invalid_json_strips_quotes(self):
        # startswith/endswith '"' but not valid JSON -> quote-strip fallback
        wrapped = '"print(\'hello\')" + "'
        result = _clean_code_string(wrapped)
        assert result == wrapped[1:-1]

    def test_plain_code_passes_through(self):
        code = 'x = 1\nprint(x)'
        assert _clean_code_string(code) == code


class TestDeeplyNestedJson:

    def test_deeply_nested_json_fails_softly(self):
        # Decoder recursion thresholds differ between Python versions.
        depth = sys.getrecursionlimit() * 2
        deeply_nested = '[' * depth + '1' + ']' * depth
        # Must not raise: RecursionError is handled at the parsing boundary.
        result = _clean_code_string(deeply_nested)
        assert isinstance(result, str)

    def test_try_json_loads_catches_recursion_error(self, monkeypatch):
        def too_deep(*args, **kwargs):
            raise RecursionError('decoder recursion limit')

        monkeypatch.setattr(python_executor.json, 'loads', too_deep)
        ok, value = _try_json_loads('[1]')
        assert ok is False and value is None


class TestNestedCodeFields:

    def test_single_wrapped_code_field(self):
        payload = json.dumps({'code': 'print(42)'})
        assert _clean_code_string(payload) == 'print(42)'

    def test_object_nested_code_field(self):
        # Object nesting is unwrapped by _parse_code_parameter in one pass.
        payload = json.dumps({'code': {'code': 'print(42)'}})
        assert _clean_code_string(payload) == 'print(42)'


class TestDepthLimit:

    def test_depth_limit_terminates_endless_chain(self, monkeypatch):
        # A parser result that always produces a fresh wrapped payload would
        # recurse forever without the limit; with it, the call terminates
        # after a deterministic number of unwraps.
        calls = {'n': 0}

        def endless(parsed, **kwargs):
            calls['n'] += 1
            return json.dumps({'code': 'level%d' % calls['n']})

        monkeypatch.setattr(python_executor, '_parse_code_parameter', endless)
        result = _clean_code_string(json.dumps({'code': 'x'}))
        assert isinstance(result, str)
        assert calls['n'] <= _MAX_CLEAN_DEPTH

    def test_at_limit_returns_input(self):
        assert _clean_code_string('anything', _depth=_MAX_CLEAN_DEPTH) == 'anything'

    def test_dictionary_nesting_beyond_limit_is_not_unwrapped(self):
        payload = 'print(42)'
        for _ in range(_MAX_CLEAN_DEPTH + 1):
            payload = {'code': payload}
        encoded = json.dumps(payload)
        assert _clean_code_string(encoded) == encoded
        with pytest.raises(ValueError, match='nesting'):
            _parse_code_parameter(payload)

    def test_dictionary_nesting_at_limit_still_extracts_code(self):
        payload = 'print(42)'
        for _ in range(_MAX_CLEAN_DEPTH):
            payload = {'code': payload}
        assert _parse_code_parameter(payload) == 'print(42)'
        assert _clean_code_string(json.dumps(payload)) == 'print(42)'

    def test_dictionary_limit_respects_existing_cleanup_depth(self):
        encoded = json.dumps({'code': {'code': 'print(42)'}})
        assert _clean_code_string(encoded, _depth=_MAX_CLEAN_DEPTH - 1) == encoded

    @pytest.mark.parametrize('key', ['code', 'unknown'])
    def test_cyclic_dictionary_fails_deterministically(self, key):
        payload = {}
        payload[key] = payload
        with pytest.raises(ValueError, match='nesting|cyclic'):
            _parse_code_parameter(payload)


class TestProcessControlPropagation:

    def test_keyboard_interrupt_propagates(self, monkeypatch):
        def raise_interrupt(*args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(python_executor.json, 'loads', raise_interrupt)
        with pytest.raises(KeyboardInterrupt):
            _clean_code_string('{"code": "x"}')

    def test_system_exit_propagates(self, monkeypatch):
        def raise_exit(*args, **kwargs):
            raise SystemExit(3)

        monkeypatch.setattr(python_executor.json, 'loads', raise_exit)
        with pytest.raises(SystemExit):
            _clean_code_string('{"code": "x"}')

    def test_unrelated_cleanup_exception_propagates(self, monkeypatch):
        # Errors raised by the cleanup logic itself (outside the parsing
        # boundary) must not be swallowed.
        def boom(*args, **kwargs):
            raise ValueError('cleanup bug')

        monkeypatch.setattr(python_executor, '_parse_code_parameter', boom)
        with pytest.raises(ValueError, match='cleanup bug'):
            _clean_code_string(json.dumps({'code': 'x'}))
