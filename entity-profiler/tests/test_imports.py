def test_basic_imports():
    import entity_profiler
    from entity_profiler.api.main import app

    assert app is not None
