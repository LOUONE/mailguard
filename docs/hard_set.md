# Hard evaluation set

The agent only costs something on the emails it is sent, so it is judged on those: test emails
the classifier is unsure of or wrong about, labelled by hand. This page is the annotation
guideline and the record of how the set was built. Code: `mailguard.agent.hard_set`, driven by
`scripts/hard_set.py`.

What is committed is [`evals/hard_set.csv`](../evals/hard_set.csv): message ids, duplicate
clusters, why each was selected, the hand label and one rationale category. No message content:
it is rebuilt from the downloaded corpora (`load_hard_set` fails if an id is missing).

## Selection

| | |
|---|---|
| Model | `mailguard-lightgbm` v7, the `production` alias (v0.2.0) |
| Split | test only (2,113 messages, 1,670 clusters) |
| Grey band | calibrated score in [0.03, 0.97] |
| Errors | wrong at the 1% operating point (threshold 0.2908, fixed on validation ham) |
| One per cluster | the qualifying message with the smallest id |
| Result | **143 messages**: 119 grey only, 20 grey and wrong, 4 wrong with a confident score |

- **Why such a wide band.** The routing grey zone is chosen later (#49). Any zone inside
  [0.03, 0.97] is a subset of this set, so each one tried can be evaluated without labelling
  more. A strict [0.1, 0.9] would give 67 clusters, too few for intervals.
- **Why one per cluster.** Copies of one campaign would count one hard case several times.
- **Why test only.** The grey-zone thresholds will be chosen from validation (routing rate and
  cost). Choosing them from the agent's accuracy on this set and then reporting that accuracy
  would be selection on the test set.
- **The 4 confident errors** are misses or false alarms the classifier was sure about. A grey
  zone never routes them, so they measure what the agent cannot fix, not what it fixes.

## Labels

Judge the message as its recipient would read it, not by the corpus it came from.

| Label | The message... |
|---|---|
| `phishing` | tries, by deception, to make the reader act against their own interest: give credentials, pay, send personal data, call a number, open a file. It usually pretends to be someone trusted. |
| `spam` | advertises something real, even shady (pharmacy, casino, cheap loans), without pretending to be someone else and without a credential, payment or data trick. |
| `ham` | is mail the recipient wanted or expected: from a person, a mailing list, a newsletter they subscribed to, an automated notice from a real service. |
| `unclear` | cannot be judged after reading it: empty, undecodable, image only, or truly balanced. Keep it rare (under 10%). |

Decision rules, in order:

1. **Impersonation plus a request is phishing**, however clumsy. A badly spelled "PayPal" asking
   to "verify your account" is `phishing/credential`.
2. **Advance-fee fraud is phishing** (`phishing/advance_fee`): lottery wins, inheritances, 419
   letters. Money or data is asked for under false pretences. ADR 0001 does not settle it and
   the error analysis left it ambiguous, so it has its own category: #49 reports the agent with
   and without it.
3. **A real company's own notice is ham**, even about accounts or payments (`ham/automated`),
   unless something says it is not theirs (a lookalike domain, a link to elsewhere).
4. **Spam versus phishing**: ask whether it impersonates someone or asks for credentials,
   payment or personal data. No: `spam`. Yes: `phishing`.
5. **Language does not decide**: a German credential lure is phishing.
6. **Padding does not decide**: a lure under paragraphs of unrelated text is phishing.

Never click a link, open an attachment or look a domain up while annotating. Links are shown
defanged (`hxxps://`, `example[.]com`), attachments as metadata only.

## Categories

| Label | Category | For |
|---|---|---|
| phishing | `credential` | fake login, account verification, suspension, mailbox quota |
| | `payment` | invoice, billing update, gift card, wire transfer |
| | `personal_data` | identity documents, tax refund forms, data-harvesting surveys |
| | `callback` | call a phone number: fake support, fake order to cancel |
| | `malware` | an attachment or download to open |
| | `advance_fee` | lottery, inheritance, 419: pay first, reward never |
| | `other_phishing` | anything else that fits the label |
| spam | `marketing` | products, services, pharmacy, casino, adult |
| | `other_spam` | |
| ham | `personal` | from one person to another, work included |
| | `list_or_newsletter` | mailing lists, subscribed newsletters |
| | `automated` | receipts, notifications, bounces from a real service |
| | `other_ham` | |
| unclear | `unreadable` | empty, undecodable, image or attachment only |
| | `ambiguous` | readable, but either label is defensible |

## Annotating

```
uv run python scripts/hard_set.py annotate
```

Messages come one by one, in a fixed random order, **without their id, corpus label or score**:
the id names the corpus file, which gives the label away. Keys: `p` `s` `h` `u` for the label,
then the category number; `m` shows the whole body, `k` skips, `q` stops. Every answer is saved
at once to `data/hard_set/annotations.csv` (gitignored), so a session can stop anywhere and
resume. `status` shows progress; a wrong answer is fixed by editing that file.

The blindness is partial: some headers still carry corpus traces (2002 SpamAssassin addresses,
the phishing corpus's recipient), and only the number of recipients is shown for that reason.

About a minute per message, so two or three sessions. Then:

```
uv run python scripts/hard_set.py export
```

writes `evals/hard_set.csv` and prints the corpus labels against the hand labels. That table,
and the category counts, go below.

## Result

To be filled after annotation: agreement with the corpus labels, label and category counts.
