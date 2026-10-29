"""Hypothesis strategies for raw email, shared by the parser and the API fuzz tests."""

from __future__ import annotations

from hypothesis import strategies as st

# Building blocks that look like email, so the search reaches deep code paths
# (MIME, encoded words, charsets) instead of stopping at "no headers found".
_header_names = st.sampled_from(
    ["From", "To", "Reply-To", "Return-Path", "Subject", "Date", "Message-ID", "Received",
     "Content-Type", "Content-Transfer-Encoding", "Content-Disposition", "MIME-Version"]
)  # fmt: skip
_header_values = st.one_of(
    st.binary(max_size=80),
    st.sampled_from(
        [
            b"=?utf-8?b?w6k=?=",
            b"=?bogus?q?abc?=",
            b"=?utf-8?b?###?=",
            b"multipart/mixed; boundary=B",
            b"multipart/alternative",
            b"text/html; charset=x-unknown",
            b"text/plain; charset=base64",
            b"base64",
            b"quoted-printable",
            b'attachment; filename="=?utf-8?q?f=C3=A9.pdf?="',
            b"<a@b.c>, , ,<",
            b"Mon, 1 Jan 99999 00:00:00 +0000",
        ]
    ),
)
_header_lines = st.lists(
    st.tuples(_header_names, _header_values).map(lambda kv: kv[0].encode() + b": " + kv[1]),
    max_size=12,
)
_body = st.one_of(
    st.binary(max_size=400),
    st.sampled_from([b"--B\n\nx\n--B--\n", b"--B\nContent-Type: text/html\n\n<a href=x>\n"]),
)
email_like = st.builds(
    lambda headers, sep, body: b"\n".join(headers) + sep + body,
    _header_lines,
    st.sampled_from([b"\n\n", b"\r\n\r\n", b"\n", b""]),
    _body,
)
