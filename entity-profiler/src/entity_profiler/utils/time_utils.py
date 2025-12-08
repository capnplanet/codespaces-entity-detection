import datetime as _dt


def hour_of_day(timestamp: float) -> int:
    """Return hour of day (0-23) for a POSIX timestamp."""
    return _dt.datetime.utcfromtimestamp(timestamp).hour
