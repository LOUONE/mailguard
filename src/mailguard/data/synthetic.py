"""Made-up emails: demo traffic for the stack, and a probe of how the model treats personal mail.

The ham corpus is 75% mailing-list mail (issue #73), so the test split says little about short
personal messages, the most common legitimate mail in a real inbox. These generators fill that
gap with obviously synthetic messages: RFC 2606 domains, documentation IP ranges, no real
person or brand account. They are a probe, not a dataset: English, short, and only as varied as
the templates below; a rate measured on them is a symptom, not an FPR.
"""

from __future__ import annotations

import random
from typing import Final

NAMES: Final = ("alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi", "ivan", "judy")
TOPICS: Final = (
    ("Lunch on Thursday?", "Are we still on for lunch on Thursday? I can bring the slides."),
    ("Re: quarterly report", "Thanks, the figures look right. I left two comments in section 3."),
    ("Team offsite", "The offsite moves to the 14th. Please confirm your dietary needs."),
    ("Photos from the weekend", "Here are the photos from Saturday. The one by the lake is great."),
    ("Dinner plans", "Shall we try the new place near the station on Friday around eight?"),
    ("Re: draft contract", "I read the draft. Clause 4 needs a date, otherwise it looks fine."),
    ("Moving day", "We move on the 3rd. Could you help with the boxes in the morning?"),
    ("Book recommendation", "I finished the novel you lent me. The last chapter surprised me."),
    ("Meeting notes", "Notes from today: budget approved, hiring starts next month, demo Tuesday."),
    ("Re: holiday dates", "I can take the week of the 20th. Does that work for the team?"),
    ("Running late", "Train is delayed, I will be about twenty minutes late. Start without me."),
    ("Birthday present", "Any idea what to get Sam for her birthday? She mentioned a new bike."),
)
FILLER: Final = (
    "Let me know what you think.",
    "No rush on this.",
    "Talk soon.",
    "Thanks again for yesterday.",
    "I attached nothing, it is all in this message.",
    "Say hi to everyone from me.",
)
LURES: Final = (
    ("Urgent: verify your account", "Unusual activity was detected. Confirm your password at"),
    ("Your mailbox is full", "You have 5 pending messages. Upgrade your quota now at"),
    ("Payment declined", "We could not process your last payment. Update your billing at"),
    ("Invoice #{n} attached", "Please review the attached invoice and sign in to view it at"),
)
BRANDS: Final = ("PayPal", "Microsoft", "DHL", "Netflix", "Your Bank")


def personal(rng: random.Random) -> str:
    """A short personal or work email between two people, without any list header."""
    sender, recipient = rng.sample(NAMES, 2)
    subject, body = rng.choice(TOPICS)
    extra = " ".join(rng.sample(FILLER, rng.randint(0, 2)))
    quoted = ""
    if subject.startswith("Re:") and rng.random() < 0.7:
        quoted = f"\n\nOn Monday, {recipient.title()} wrote:\n> Any news on this?\n> Thanks"
    signature = rng.choice(("", f"\n\n-- \n{sender.title()}", f"\n\nBest,\n{sender.title()}"))
    return (
        f"Return-Path: <{sender}@example.org>\nFrom: {sender.title()} <{sender}@example.org>\n"
        f"To: {recipient}@example.org\nSubject: {subject}\n"
        f"Message-ID: <{rng.getrandbits(48):x}@example.org>\n"
        f"Date: Thu, 29 Oct 2026 10:00:00 +0000\n\n"
        f"Hi {recipient.title()},\n\n{body} {extra}".rstrip()
        + f"{signature}{quoted}\n"
    )


def phishing(rng: random.Random) -> str:
    """A credential lure from a brand-like sender, with a link to a bare IP."""
    brand = rng.choice(BRANDS)
    subject, lure = rng.choice(LURES)
    host = f"198.51.100.{rng.randint(1, 254)}"
    return (
        f'From: "{brand} Security" <no-reply@secure-{brand.lower().replace(" ", "")}.example>\n'
        f"Reply-To: support@other.example\nSubject: {subject.format(n=rng.randint(1000, 9999))}\n"
        f"Content-Type: text/html\n\n<p>Dear customer,</p><p>{lure} "
        f'<a href="http://{host}/login.php?id={rng.getrandbits(32):x}">www.{brand.lower()}.com</a>'
        "</p>\n"
    )


def personal_probe(n: int = 300, seed: int = 0) -> list[bytes]:
    rng = random.Random(seed)
    return [personal(rng).encode() for _ in range(n)]
