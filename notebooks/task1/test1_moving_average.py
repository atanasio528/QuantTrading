import csp
from csp import ts, stats
import numpy as np
from datetime import datetime, timedelta
import pytest
import matplotlib.pyplot as plt
import pandas as pd

st = datetime(2025, 1, 1)
en = st + timedelta(seconds=5)
price = csp.curve(
    typ=float,
    data=[
        (st + timedelta(seconds=1.0), 12.47),
        (st + timedelta(seconds=1.2), 11.89),
        (st + timedelta(seconds=1.4), 10.23),
        (st + timedelta(seconds=1.6), 13.78),
        (st + timedelta(seconds=1.8), 9.95),
        (st + timedelta(seconds=2.0), 12.66),
        (st + timedelta(seconds=2.2), 13.01),
        (st + timedelta(seconds=2.4), 11.22),
        (st + timedelta(seconds=2.6), 10.75),
        (st + timedelta(seconds=2.8), 12.38),
        (st + timedelta(seconds=3.0), 9.88),
        (st + timedelta(seconds=3.2), 10.54),
        (st + timedelta(seconds=3.4), 11.67),
        (st + timedelta(seconds=3.6), 13.94),
        (st + timedelta(seconds=3.8), 12.15),
        (st + timedelta(seconds=4.0), 10.06),
        (st + timedelta(seconds=4.2), 9.73),
        (st + timedelta(seconds=4.4), 11.98),
        (st + timedelta(seconds=4.6), 13.26),
        (st + timedelta(seconds=4.8), 12.34),
        (st + timedelta(seconds=5.0), 14.14),
    ]
)

@csp.node
def time_moving_average(x: ts[float], window_size: float) -> ts[float]:
    with csp.start():
        # TODO
        assert window_size > 0
        csp.set_buffering_policy(x, tick_history=timedelta(seconds=window_size))

    if csp.ticked(x):
        # TODO
        buffer = csp.values_at(
            x,
            start_index_or_time=timedelta(seconds=-window_size),
            end_index_or_time=timedelta(seconds=0),
            start_index_policy=csp.TimeIndexPolicy.EXCLUSIVE, # Caution!: exclude start time to match # of ticks
            end_index_policy=csp.TimeIndexPolicy.INCLUSIVE)
        if len(buffer) > 0:
            return np.mean(buffer)


@csp.node
def tick_moving_average(x: ts[float], tick_count: int) -> ts[float]:
    with csp.start():
        # TODO
        assert tick_count > 0
        csp.set_buffering_policy(x, tick_count=tick_count)
    if csp.ticked(x):
        # TODO
        buffer = csp.values_at(
            x,
            start_index_or_time=- (tick_count - 1),  # Caution!: subtract one for indexing
            end_index_or_time=0
        )
        if len(buffer) > 0:
            return np.mean(buffer)

@csp.graph
def time_ma_graph(x: ts[float], window_size: float):
    time_ma = time_moving_average(x, window_size)
    csp_time_ma = stats.mean(
        x,
        interval=timedelta(seconds=window_size),
        min_window=timedelta(seconds=0),
        min_data_points=0,
    )
    csp.add_graph_output("my_time_ma", time_ma)
    csp.add_graph_output("csp_time_ma", csp_time_ma)


@csp.graph
def tick_ma_graph(x: ts[float], tick_count: int):
    tick_ma = tick_moving_average(x, tick_count)
    csp_tick_ma = stats.mean(
        x,
        interval=tick_count,
        min_window=0,
        min_data_points=0,
    )
    csp.add_graph_output("my_tick_ma", tick_ma)
    csp.add_graph_output("csp_tick_ma", csp_tick_ma)


@csp.graph
def my_ma_graph(x: ts[float], window_size: float, tick_count: int):
    time_ma = time_moving_average(x, window_size)
    tick_ma = tick_moving_average(x, tick_count)
    csp.add_graph_output("my_time_ma", time_ma)
    csp.add_graph_output("my_tick_ma", tick_ma)


@pytest.mark.parametrize("window_size", [1, 1.5])
def test_time_ma(window_size):
    results = csp.run(
        time_ma_graph,
        x=price,
        window_size=window_size,
        starttime=st,
        endtime=en
    )

    my_time_ma = results["my_time_ma"]
    csp_time_ma = results["csp_time_ma"]

    assert len(my_time_ma) == len(csp_time_ma)
    for (t1, v1), (t2, v2) in zip(my_time_ma, csp_time_ma):
        assert t1 == t2
        assert np.isclose(v1, v2)

@pytest.mark.parametrize("tick_count", [5])
def test_tick_ma(tick_count):
    results = csp.run(
        tick_ma_graph,
        x=price,
        tick_count=tick_count,
        starttime=st,
        endtime=en,
    )

    my_tick_ma = results["my_tick_ma"]
    csp_tick_ma = results["csp_tick_ma"]

    assert len(my_tick_ma) == len(csp_tick_ma)
    for (t1, v1), (t2, v2) in zip(my_tick_ma, csp_tick_ma):
        assert t1 == t2
        assert np.isclose(v1, v2)

testset3 = [(1, 5), (1.2, 6), (1.4, 7), (1.6, 8), (1.8, 9), (2, 10)]
@pytest.mark.parametrize("window_size, tick_count", testset3)
# Given the price data, we can assume 1 second = 5 ticks
def test_my_ma(window_size, tick_count):
    results = csp.run(
        my_ma_graph,
        starttime=st,
        endtime=en,
        x=price,
        window_size=window_size,
        tick_count=tick_count)

    my_time_ma = results["my_time_ma"]
    my_tick_ma = results["my_tick_ma"]

    assert len(my_time_ma) == len(my_tick_ma)
    for (t1, v1), (t2, v2) in zip(my_time_ma, my_tick_ma):
        assert t1 == t2
        assert np.isclose(v1, v2), f"Mismatch at {t1}: time_ma={v1}, tick_ma={v2}"
