"""Cycle sampling, cycle window, unknown-value check and the cycle-state hex format."""
import hashlib
import random

import pytest

from setfi.workload import sampling as S
from setfi.workload.errors import WorkloadError

# Reference vectors (cycles 1..19200 after reset, padding 5, seed 12345, 30 samples,
# reserve factor 3).
PRIMARY_19200 = [339, 2556, 2997, 4077, 4860, 5304, 5464, 5718, 6127, 6230, 6351, 6746,
                 8561, 8861, 9791, 10558, 11198, 11635, 12078, 12230,
                 13510, 13657, 14192, 14300, 16518, 17407, 18133,
                 18424, 18552, 19176]
RESERVE_19200 = [110, 115, 119, 224, 312, 390, 427, 459, 777, 887, 1325, 1368, 1623,
                 2002, 2023, 2582, 2630, 2976, 3150, 3269, 3478, 3920,
                 3934, 3990, 4265, 4811, 4915, 4930, 4935, 5154, 5345,
                 5370, 5379, 5390, 5417, 5875, 5895, 5966, 6071, 6103,
                 6622, 6677, 7074, 7108, 7525, 7575, 7630, 7640, 8463,
                 8678, 9410, 9466, 9894, 10630, 10696, 10831, 11017,
                 11147, 11228, 11482, 11960, 12074, 12314, 12674,
                 12743, 13334, 13610, 13641, 13817, 14239, 14615,
                 15065, 15784, 16145, 16246, 16285, 16388, 16584,
                 16765, 16825, 17012, 17057, 17188, 17632, 17698,
                 18048, 18222, 18270, 18977, 19044]


# ---------------------------------------------------------------- sampling
def test_sample_cycles_pinned_vector():
    p, r = S.sample_cycles(0, 19200, 30, 5, 12345)
    assert p == PRIMARY_19200
    assert r == RESERVE_19200


def test_sample_cycles_pinned_small():
    assert S.sample_cycles(0, 2400, 5, 5, 1, reserve_factor=2) == (
        [264, 488, 556, 1050, 2337], [122, 391, 868, 1564, 1606, 1782, 1851, 1944, 2008, 2039])


def test_sample_cycles_deterministic_and_disjoint():
    a = S.sample_cycles(3, 5000, 30, 5, 99, warmup=10)
    b = S.sample_cycles(3, 5000, 30, 5, 99, warmup=10)
    assert a == b
    p, r = a
    assert p == sorted(p) and r == sorted(r)
    assert not set(p) & set(r)
    assert len(r) == 90
    assert all(19 <= c <= 4995 for c in p + r)          # 3 + 1 + 5 padding + 10 warmup


def test_primary_does_not_depend_on_reserve_factor():
    # the primary draw is the first use of the RNG stream
    base = S.sample_cycles(0, 10000, 30, 5, 7, reserve_factor=0)[0]
    for rf in (1, 3, 10):
        assert S.sample_cycles(0, 10000, 30, 5, 7, reserve_factor=rf)[0] == base
    assert S.sample_cycles(0, 10000, 30, 5, 7, reserve_factor=0)[1] == []


def test_reserve_capped_by_pool():
    p, r = S.sample_cycles(0, 50, 30, 5, 1)
    assert len(p) == 30 and len(r) == 40 - 30   # pool [6, 45]


def test_sample_cycles_range_too_small():
    with pytest.raises(WorkloadError) as e:
        S.sample_cycles(0, 30, 30, 5, 1)
    assert e.value.kind == "no_cycles_in_range"


# ---------------------------------------------------------------- cycle window
def _write_csv(tmp_path, rows, header="start_cycle,end_cycle"):
    p = tmp_path / "win.csv"
    p.write_text(header + "\n" + "".join(f"{a},{b}\n" for a, b in rows))
    return p


def test_load_cycle_window(tmp_path):
    p = _write_csv(tmp_path, [(10, 12), (20, 20), (11, 15)])
    cycles, meta = S.load_cycle_window(p)
    assert cycles == {10, 11, 12, 13, 14, 15, 20}
    assert meta["n_rows"] == 3 and meta["n_cycles"] == 7
    assert meta["min_cycle"] == 10 and meta["max_cycle"] == 20
    assert meta["md5"] == hashlib.md5(p.read_bytes()).hexdigest()


@pytest.mark.parametrize("rows,header", [
    ([(5, 3)], "start_cycle,end_cycle"),          # start > end
    ([("a", 3)], "start_cycle,end_cycle"),        # not an integer
    ([(1, 3)], "start,end"),                      # wrong header
    ([], "start_cycle,end_cycle"),                # no cycle
])
def test_load_cycle_window_rejects(tmp_path, rows, header):
    p = _write_csv(tmp_path, rows, header)
    with pytest.raises(WorkloadError) as e:
        S.load_cycle_window(p)
    assert e.value.kind == "config_invalid"


def test_window_sampler_restricts_pool():
    window = set(range(100, 200)) | {7, 3000}
    p, r, n_pool = S.sample_cycles_in_window(0, 2400, 20, 5, 42, window)
    assert n_pool == 101                     # 3000 is outside the working range
    assert set(p) | set(r) <= window
    assert not set(p) & set(r)
    assert S.sample_cycles_in_window(0, 2400, 20, 5, 42, window) == (p, r, n_pool)


def test_window_sampler_whole_pool():
    window = {59, 189, 286}
    p, r, n = S.sample_cycles_in_window(0, 2400, 3, 5, 12345, window)
    assert p == [59, 189, 286] and r == [] and n == 3


def test_window_sampler_too_small():
    with pytest.raises(WorkloadError) as e:
        S.sample_cycles_in_window(0, 2400, 5, 5, 1, {1, 2, 100, 200})  # 1, 2 below lo=5
    assert e.value.kind == "no_cycles_in_range"


# ---------------------------------------------------------------- unknown-value check
def test_x_frac_of_text():
    assert S.x_frac_of_text("00ff\n") == 0.0
    assert S.x_frac_of_text("0x0X") == 0.5
    assert S.x_frac_of_text("zZ") == 1.0
    assert S.x_frac_of_text("") == 1.0
    assert S.x_frac_of_text("ABCDEF") == 0.0


def test_screen_keeps_clean_primary(tmp_path):
    for c in (1, 2, 3, 4):
        (tmp_path / f"cycle_{c}.hex").write_text("0" * 100 + "\n")
    sel, rep = S.screen_cycles(tmp_path, [1, 2], [3, 4], 2)
    assert sel == [1, 2]
    assert rep["replaced_cycles"] == [] and rep["replacement_cycles"] == []
    assert rep["n_candidates_recorded"] == 4
    assert rep["x_fraction_max"] == 0.0


def test_screen_backfills_in_reserve_order(tmp_path):
    dirty = "x" * 3 + "0" * 97          # 3 % unresolved > 2 %
    edge = "x" * 2 + "0" * 98           # exactly 2 %: kept
    files = {10: dirty, 20: "0" * 100, 30: dirty, 5: dirty, 6: edge, 7: "0" * 100, 8: "0" * 100}
    for c, t in files.items():
        (tmp_path / f"cycle_{c}.hex").write_text(t + "\n")
    logs = []
    sel, rep = S.screen_cycles(tmp_path, [10, 20, 30], [5, 6, 7, 8], 3, log=logs.append)
    assert rep["replaced_cycles"] == [10, 30]
    assert rep["replacement_cycles"] == [6, 7]      # 5 is dirty, 8 not needed
    assert sel == [6, 7, 20]
    assert rep["x_fraction_max"] == 0.02
    assert any("2/3 sampled cycles replaced" in m for m in logs)


def test_screen_threshold_parameter(tmp_path):
    (tmp_path / "cycle_1.hex").write_text("x" * 3 + "0" * 97 + "\n")
    sel, _ = S.screen_cycles(tmp_path, [1], [], 1, max_x_frac=0.05)
    assert sel == [1]
    with pytest.raises(WorkloadError) as e:
        S.screen_cycles(tmp_path, [1], [], 1)
    assert e.value.kind == "no_cycles_in_range"


def test_screen_missing_file_counts_unresolved():
    sel, rep = S.screen_cycles_by_frac({2: 0.0}, [1], [2], 1)
    assert sel == [2] and rep["replaced_cycles"] == [1]


# ---------------------------------------------------------------- hex format
def test_format_state_hex_basic():
    # net 0 is the LSB
    assert S.format_state_hex([1, 0, 0, 0]) == "1"
    assert S.format_state_hex([0, 0, 0, 0, 1]) == "10"
    assert S.format_state_hex([1, 1, 1, 1, 0, 1, 0, 1]) == "af"
    assert S.format_state_hex([1, 1, 1]) == "7"         # partial top digit


def test_format_state_hex_four_state_rules():
    assert S.format_state_hex("xxxx") == "x"
    assert S.format_state_hex("x000") == "X"
    assert S.format_state_hex("zzzz") == "z"
    assert S.format_state_hex("z100") == "Z"
    assert S.format_state_hex("zx00") == "X"            # x wins over z
    assert S.format_state_hex("zzzx") == "X"
    assert S.format_state_hex(["1", "1", "x"]) == "X"   # 3-bit top digit
    assert S.format_state_hex(["x", "x", "x"]) == "x"


def test_state_hex_roundtrip():
    rng = random.Random(3)
    for n in (1, 3, 4, 5, 63, 64, 65, 3787):
        bits = [rng.choice("01") for _ in range(n)]
        text = S.format_state_hex(bits)
        assert len(text) == S.n_hex_digits(n)
        assert S.parse_state_hex(text, n) == bits
        assert int(text, 16) == sum(1 << i for i, b in enumerate(bits) if b == "1")


def test_parse_state_hex_length_check():
    with pytest.raises(ValueError):
        S.parse_state_hex("abc", 16)
