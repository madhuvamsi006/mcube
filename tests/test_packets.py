"""Binary packet parser tests.

The parser runs on every tick of every session. A silent offset error here
corrupts an entire day of unrecoverable data, so each packet type is pinned
with a hand-built frame.
"""

import struct

import pytest

from mcube.collector.packets import (
    CODE_FULL,
    CODE_OI,
    CODE_QUOTE,
    CODE_TICKER,
    HEADER_SIZE,
    parse_header,
    parse_stream,
    subscribe_message,
)


def _header(code: int, length: int, segment: int, sec_id: int) -> bytes:
    return struct.pack("<BhBi", code, length, segment, sec_id)


def test_header_offsets():
    raw = _header(CODE_FULL, 162, 2, 61471)
    h = parse_header(raw)
    assert (h.code, h.length, h.segment, h.security_id) == (CODE_FULL, 162, 2, 61471)
    assert h.segment_name == "NSE_FNO"


def test_ticker_packet():
    body = struct.pack("<fi", 26000.5, 1793720180)
    raw = _header(CODE_TICKER, HEADER_SIZE + len(body), 0, 13) + body
    tick = next(iter(parse_stream(raw)))
    assert tick.security_id == 13
    assert tick.ltp == pytest.approx(26000.5, rel=1e-6)
    assert tick.ltt == 1793720180


def test_oi_packet():
    body = struct.pack("<i", 2117180)
    raw = _header(CODE_OI, HEADER_SIZE + len(body), 2, 61471) + body
    tick = next(iter(parse_stream(raw)))
    assert tick.oi == 2117180


def test_quote_packet_field_order():
    """Pins the exact field order -- a swap here is silent and corrupting."""
    body = struct.pack(
        "<fhifiiiffff",
        26000.5,      # ltp
        75,           # ltq
        1793720180,   # ltt
        25990.25,     # atp
        1234567,      # volume
        5000,         # total sell qty
        6000,         # total buy qty
        25900.0,      # day open
        25950.0,      # day close
        26100.0,      # day high
        25880.0,      # day low
    )
    raw = _header(CODE_QUOTE, HEADER_SIZE + len(body), 2, 61471) + body
    t = next(iter(parse_stream(raw)))
    assert t.ltp == pytest.approx(26000.5, rel=1e-6)
    assert t.ltq == 75
    assert t.volume == 1234567
    assert t.total_buy_qty == 6000
    assert t.total_sell_qty == 5000
    assert t.day_high == pytest.approx(26100.0, rel=1e-6)
    assert t.day_low == pytest.approx(25880.0, rel=1e-6)


def _full_packet(sec_id: int = 61471) -> bytes:
    quote = struct.pack(
        "<fhifiiiffff",
        26000.5, 75, 1793720180, 25990.25, 1234567,
        5000, 6000, 25900.0, 25950.0, 26100.0, 25880.0,
    )
    extra = struct.pack("<fff", 25950.0, 26100.0, 25880.0)
    depth = b""
    for lv in range(5):
        depth += struct.pack(
            "<iihhff",
            100 * (lv + 1),              # bid qty
            200 * (lv + 1),              # ask qty
            lv + 1,                      # bid orders
            lv + 2,                      # ask orders
            26000.0 - 0.05 * lv,         # bid price
            26000.5 + 0.05 * lv,         # ask price
        )
    body = quote + extra + depth
    return _header(CODE_FULL, HEADER_SIZE + len(body), 2, sec_id) + body


def test_full_packet_depth_ladder():
    t = next(iter(parse_stream(_full_packet())))
    assert len(t.bid_px) == 5 and len(t.ask_px) == 5
    # Bids descend, asks ascend -- a transposed ladder would break this.
    assert t.bid_px == sorted(t.bid_px, reverse=True)
    assert t.ask_px == sorted(t.ask_px)
    assert t.bid_qty[0] == 100 and t.ask_qty[0] == 200
    assert t.bid_px[0] > 0 and t.ask_px[0] > t.bid_px[0]


def test_multiple_packets_in_one_frame():
    """A single TCP read often carries several packets concatenated."""
    frame = _full_packet(61471) + _full_packet(61466) + _full_packet(13)
    ticks = list(parse_stream(frame))
    assert [t.security_id for t in ticks] == [61471, 61466, 13]


def test_truncated_frame_does_not_hang_or_crash():
    frame = _full_packet()[:40]
    assert list(parse_stream(frame)) == []


def test_zero_length_header_terminates():
    """A malformed zero length would loop forever if not guarded."""
    raw = _header(CODE_FULL, 0, 2, 61471) + b"\x00" * 50
    assert list(parse_stream(raw)) == []


def test_to_row_has_all_depth_columns():
    t = next(iter(parse_stream(_full_packet())))
    row = t.to_row(recv_ns=1)
    for lv in range(1, 6):
        for side in ("bid", "ask"):
            assert f"{side}_px_{lv}" in row
            assert f"{side}_qty_{lv}" in row
    assert row["security_id"] == 61471


def test_subscribe_message_shape():
    msg = subscribe_message([("NSE_FNO", "61471"), ("IDX_I", "13")], request_code=21)
    assert msg["RequestCode"] == 21
    assert msg["InstrumentCount"] == 2
    assert msg["InstrumentList"][0] == {"ExchangeSegment": "NSE_FNO", "SecurityId": "61471"}
