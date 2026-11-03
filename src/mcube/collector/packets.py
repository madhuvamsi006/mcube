"""Binary packet parsing for the DhanHQ v2 live market feed.

Wire format (DhanHQ v2 docs, little-endian throughout):

  Header, 8 bytes, on every packet:
    offset 0      uint8    feed response code
    offset 1..2   int16    message length
    offset 3      uint8    exchange segment
    offset 4..7   int32    security id

  Response codes:
    2   ticker   LTP + last trade time
    4   quote    LTP, LTQ, ATP, volume, buy/sell qty, OHLC
    5   OI       open interest
    8   full     quote fields + 5-level market depth
    50  disconnect

  Full packet, 162 bytes: quote fields then 5 depth levels of 20 bytes each:
    int32 bid_qty | int32 ask_qty | int16 bid_orders | int16 ask_orders
    float32 bid_price | float32 ask_price

  200-level depth arrives on a SEPARATE socket with its own 12-byte header
  and 16-byte packets (float64 price, uint32 qty, uint32 orders), bids and
  asks in separate messages.

A single TCP read can contain several concatenated packets, so `parse_stream`
walks the buffer by each packet's declared length rather than assuming one
packet per message.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Iterator

HEADER_SIZE = 8
DEPTH_LEVEL_SIZE = 20
FULL_DEPTH_LEVELS = 5

# Feed response codes
CODE_TICKER = 2
CODE_QUOTE = 4
CODE_OI = 5
CODE_FULL = 8
CODE_DISCONNECT = 50

# Exchange segment enum -> name
SEGMENT_NAMES = {
    0: "IDX_I",
    1: "NSE_EQ",
    2: "NSE_FNO",
    3: "NSE_CURRENCY",
    4: "BSE_EQ",
    5: "MCX_COMM",
    7: "BSE_CURRENCY",
    8: "BSE_FNO",
}

_HEADER = struct.Struct("<BhBi")
# 11 fields, bytes 9..50 = 42 bytes:
#   f ltp | h ltq | i ltt | f atp | i volume | i sell_qty | i buy_qty
#   | f open | f close | f high | f low
_QUOTE_BODY = struct.Struct("<fhifiiiffff")
_DEPTH_LEVEL = struct.Struct("<iihhff")


@dataclass
class Header:
    code: int
    length: int
    segment: int
    security_id: int

    @property
    def segment_name(self) -> str:
        return SEGMENT_NAMES.get(self.segment, str(self.segment))


@dataclass
class Tick:
    """One parsed packet, flattened for storage."""

    code: int
    security_id: int
    segment: str
    ltp: float | None = None
    ltq: int | None = None
    ltt: int | None = None
    atp: float | None = None
    volume: int | None = None
    total_sell_qty: int | None = None
    total_buy_qty: int | None = None
    day_open: float | None = None
    day_close: float | None = None
    day_high: float | None = None
    day_low: float | None = None
    oi: int | None = None
    # 5 levels; index 0 is best bid/ask
    bid_px: list[float] = field(default_factory=list)
    bid_qty: list[int] = field(default_factory=list)
    bid_orders: list[int] = field(default_factory=list)
    ask_px: list[float] = field(default_factory=list)
    ask_qty: list[int] = field(default_factory=list)
    ask_orders: list[int] = field(default_factory=list)

    def to_row(self, recv_ns: int) -> dict:
        """Flatten to a storage row. recv_ns is our local receive time.

        Both the exchange timestamp (ltt) and our receive time are kept: the
        gap between them is the only way to detect a stalling feed.
        """
        row = {
            "recv_ns": recv_ns,
            "code": self.code,
            "security_id": self.security_id,
            "segment": self.segment,
            "ltp": self.ltp,
            "ltq": self.ltq,
            "ltt": self.ltt,
            "atp": self.atp,
            "volume": self.volume,
            "total_sell_qty": self.total_sell_qty,
            "total_buy_qty": self.total_buy_qty,
            "day_open": self.day_open,
            "day_high": self.day_high,
            "day_low": self.day_low,
            "day_close": self.day_close,
            "oi": self.oi,
        }
        for i in range(FULL_DEPTH_LEVELS):
            row[f"bid_px_{i+1}"] = self.bid_px[i] if i < len(self.bid_px) else None
            row[f"bid_qty_{i+1}"] = self.bid_qty[i] if i < len(self.bid_qty) else None
            row[f"bid_ord_{i+1}"] = self.bid_orders[i] if i < len(self.bid_orders) else None
            row[f"ask_px_{i+1}"] = self.ask_px[i] if i < len(self.ask_px) else None
            row[f"ask_qty_{i+1}"] = self.ask_qty[i] if i < len(self.ask_qty) else None
            row[f"ask_ord_{i+1}"] = self.ask_orders[i] if i < len(self.ask_orders) else None
        return row


def parse_header(buf: bytes, offset: int = 0) -> Header:
    code, length, segment, sec_id = _HEADER.unpack_from(buf, offset)
    return Header(code=code, length=length, segment=segment, security_id=sec_id)


def _parse_quote_body(buf: bytes, offset: int, tick: Tick) -> int:
    """Quote fields are identical in quote and full packets. Returns new offset."""
    (
        ltp, ltq, ltt, atp, volume,
        total_sell, total_buy,
        day_open, day_close, day_high, day_low,
    ) = _QUOTE_BODY.unpack_from(buf, offset)

    tick.ltp = ltp
    tick.ltq = ltq
    tick.ltt = ltt
    tick.atp = atp
    tick.volume = volume
    tick.total_sell_qty = total_sell
    tick.total_buy_qty = total_buy
    tick.day_open = day_open
    tick.day_close = day_close
    tick.day_high = day_high
    tick.day_low = day_low
    return offset + _QUOTE_BODY.size


def parse_packet(buf: bytes, offset: int, header: Header) -> Tick:
    tick = Tick(code=header.code, security_id=header.security_id,
                segment=header.segment_name)
    pos = offset + HEADER_SIZE

    if header.code == CODE_TICKER:
        ltp, ltt = struct.unpack_from("<fi", buf, pos)
        tick.ltp, tick.ltt = ltp, ltt

    elif header.code == CODE_OI:
        (tick.oi,) = struct.unpack_from("<i", buf, pos)

    elif header.code == CODE_QUOTE:
        _parse_quote_body(buf, pos, tick)

    elif header.code == CODE_FULL:
        pos = _parse_quote_body(buf, pos, tick)
        # Full packet repeats day close/high/low before the depth block.
        pos += struct.calcsize("<fff")
        for _ in range(FULL_DEPTH_LEVELS):
            if pos + DEPTH_LEVEL_SIZE > len(buf):
                break
            bq, aq, bo, ao, bp, ap = _DEPTH_LEVEL.unpack_from(buf, pos)
            tick.bid_qty.append(bq)
            tick.ask_qty.append(aq)
            tick.bid_orders.append(bo)
            tick.ask_orders.append(ao)
            tick.bid_px.append(bp)
            tick.ask_px.append(ap)
            pos += DEPTH_LEVEL_SIZE

    return tick


def parse_stream(buf: bytes) -> Iterator[Tick]:
    """Yield every packet in a websocket frame.

    Frames may carry multiple concatenated packets, so advance by the declared
    message length. A zero or negative length would loop forever -- bail out
    instead, since the stream is unparseable from that point.
    """
    offset = 0
    n = len(buf)
    while offset + HEADER_SIZE <= n:
        header = parse_header(buf, offset)

        if header.code == CODE_DISCONNECT:
            return

        if header.length <= 0 or offset + header.length > n:
            return

        yield parse_packet(buf, offset, header)
        offset += header.length


def subscribe_message(instruments: list[tuple[str, str]], request_code: int = 21) -> dict:
    """Build a subscription message.

    Request codes: 15 ticker, 17 quote, 21 full (quote + 5-level depth),
    23 full market depth, 12 disconnect.

    Dhan caps a single subscribe message at 100 instruments -- the caller is
    responsible for chunking.
    """
    return {
        "RequestCode": request_code,
        "InstrumentCount": len(instruments),
        "InstrumentList": [
            {"ExchangeSegment": seg, "SecurityId": str(sid)} for seg, sid in instruments
        ],
    }
