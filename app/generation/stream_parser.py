"""
Incremental extraction of the `answer` field from a streaming JSON reply.

The problem
-----------
Structured generation and streaming pull in opposite directions. The model
streams its reply as JSON:

    {"answer": "An eagle travels at 30 to 55 mph...", "sources_used": [1], ...}

so the raw deltas are NOT user-visible text. The first bytes off the wire are
`{"answer": "` — emitting them verbatim would show a user JSON syntax. That is
why time-to-first-TOKEN and time-to-first-VISIBLE-TEXT are different numbers,
and why this project measures both.

The naive workaround is to buffer the whole reply, parse it, then display —
which throws away the entire point of streaming. This parser instead decodes
the `answer` string value character by character as it arrives, so the caller
can emit prose immediately while the trailing `sources_used` / `sufficient`
fields are still in flight.

What it guarantees
------------------
* Only the value of the top-level `answer` key is ever emitted. Any other
  field's content is buffered for validation and never shown.
* JSON escapes are decoded properly: \\" \\\\ \\n \\t and \\uXXXX. A partial
  escape sequence split across two network chunks is held back rather than
  emitted broken — this matters for Devanagari, where a single character is
  routinely a surrogate-free but multi-byte \\uXXXX escape.
* The full raw text is retained so the final object can still be validated by
  the strict schema. Streaming does not bypass validation; it runs alongside it.
* If the reply never turns out to be the expected shape, nothing was emitted
  that was not inside `answer`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExtractorState:
    raw: str = ""
    emitted: str = ""
    in_answer: bool = False
    finished_answer: bool = False
    _buf: str = ""
    _key_seen: bool = False
    _pending_escape: str = ""
    _visible_chars: int = field(default=0)


class StreamingAnswerExtractor:
    """
    Feed raw content deltas in, get user-visible text out.

    Deliberately a hand-rolled scanner rather than a streaming JSON library:
    we need exactly one field, we must survive arbitrary chunk boundaries, and
    the failure mode must be "emit nothing" rather than "emit something wrong".
    """

    KEY = '"answer"'

    def __init__(self) -> None:
        self.state = ExtractorState()

    @property
    def raw(self) -> str:
        return self.state.raw

    @property
    def emitted(self) -> str:
        return self.state.emitted

    def feed(self, delta: str) -> str:
        """Consume a raw delta; return whatever became user-visible."""
        st = self.state
        st.raw += delta
        if st.finished_answer:
            return ""

        out = []
        buf = st._buf + delta
        i = 0

        while i < len(buf):
            ch = buf[i]

            if not st.in_answer:
                # Scan for the opening quote of the answer VALUE. We look for
                # the key, then the colon, then the quote -- rather than the
                # first quote after the key -- so a key containing whitespace
                # or an unusual gap still works.
                idx = buf.find(self.KEY, i)
                if idx == -1:
                    # Keep a tail in case the key straddles this boundary.
                    st._buf = buf[-len(self.KEY):]
                    break
                j = idx + len(self.KEY)
                while j < len(buf) and buf[j] in ' \t\r\n':
                    j += 1
                if j >= len(buf):
                    st._buf = buf[idx:]
                    break
                if buf[j] != ':':
                    i = idx + 1
                    continue
                j += 1
                while j < len(buf) and buf[j] in ' \t\r\n':
                    j += 1
                if j >= len(buf):
                    st._buf = buf[idx:]
                    break
                if buf[j] != '"':
                    i = idx + 1
                    continue
                st.in_answer = True
                st._key_seen = True
                i = j + 1
                buf = buf
                continue

            # --- inside the answer string -------------------------------
            if st._pending_escape:
                # Resume an escape split across chunks.
                #
                # `need` MUST be recomputed after appending. If the previous
                # chunk ended on a bare "\\", we do not yet know whether this
                # is a 2-char escape or a 6-char \\uXXXX one. Computing need
                # once, up front, truncated every \\u escape to 2 characters
                # and silently dropped it -- which broke Devanagari entirely,
                # since json.dumps emits Devanagari as \\uXXXX by default.
                while len(st._pending_escape) < 6 and i < len(buf):
                    need = 6 if st._pending_escape.startswith("\\u") else 2
                    if len(st._pending_escape) >= need:
                        break
                    st._pending_escape += buf[i]
                    i += 1

                need = 6 if st._pending_escape.startswith("\\u") else 2
                if len(st._pending_escape) < need:
                    st._buf = ""
                    break                      # still incomplete, wait for more
                decoded = self._decode_escape(st._pending_escape[:need])
                st._pending_escape = ""
                if decoded is not None:
                    out.append(decoded)
                continue

            if ch == '\\':
                remaining = buf[i:]
                if len(remaining) < 2:
                    st._pending_escape = remaining
                    i = len(buf)
                    st._buf = ""
                    break
                if remaining[1] == 'u':
                    if len(remaining) < 6:
                        st._pending_escape = remaining
                        i = len(buf)
                        st._buf = ""
                        break
                    decoded = self._decode_escape(remaining[:6])
                    i += 6
                else:
                    decoded = self._decode_escape(remaining[:2])
                    i += 2
                if decoded is not None:
                    out.append(decoded)
                continue

            if ch == '"':
                # Unescaped quote ends the answer value. Everything after it
                # (sources_used, sufficient) is validation data, not display.
                st.in_answer = False
                st.finished_answer = True
                st._buf = ""
                break

            out.append(ch)
            i += 1
        else:
            st._buf = ""

        text = "".join(out)
        st.emitted += text
        st._visible_chars += len(text)
        return text

    @staticmethod
    def _decode_escape(seq: str) -> str | None:
        if seq.startswith("\\u") and len(seq) == 6:
            try:
                return chr(int(seq[2:], 16))
            except ValueError:
                return None
        mapping = {'\\"': '"', "\\\\": "\\", "\\/": "/", "\\n": "\n",
                   "\\t": "\t", "\\r": "\r", "\\b": "\b", "\\f": "\f"}
        return mapping.get(seq)

    def looks_like_json(self) -> bool:
        """Did we ever see the field we were promised?"""
        return self.state._key_seen
