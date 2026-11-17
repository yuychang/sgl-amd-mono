from types import SimpleNamespace

import pytest

from sgl_amd_mono.common import compiled_launch as implementation


def test_compile_is_the_first_launch_and_runtime_arguments_stay_dynamic(monkeypatch):
    calls = []
    compiles = []

    def host(pointer, slot, stream=None):
        calls.append((pointer, slot, stream))

    jit = SimpleNamespace(func=host)

    def compile_once(function, *args, **kwargs):
        compiles.append(function)
        function.func(*args, **kwargs)
        return host

    monkeypatch.setattr(implementation.flyc, "compile", compile_once)
    launch = implementation.compiled_launch(jit)
    launch(100, slot=1, stream="warmup")
    launch(pointer=200, slot=2, stream="capture")
    launch(300, 3)
    assert calls == [(100, 1, "warmup"), (200, 2, "capture"), (300, 3, None)]
    assert compiles == [jit]


def test_compile_failure_does_not_install_a_broken_callable(monkeypatch):
    def host(pointer):
        return None

    def fail(*args, **kwargs):
        raise ValueError("compile failed")

    monkeypatch.setattr(implementation.flyc, "compile", fail)
    launch = implementation.compiled_launch(SimpleNamespace(func=host))
    with pytest.raises(ValueError, match="compile failed"):
        launch(100)
    calls = []
    monkeypatch.setattr(implementation.flyc, "compile", lambda *args: calls.append)
    launch(100)
    launch(200)
    assert calls == [200]


def test_invalid_arguments_fail_before_compilation(monkeypatch):
    def host(pointer):
        return None

    compiles = []
    monkeypatch.setattr(implementation.flyc, "compile", lambda *args: compiles.append(args))
    launch = implementation.compiled_launch(SimpleNamespace(func=host))
    with pytest.raises(TypeError):
        launch(unexpected=1)
    assert compiles == []
