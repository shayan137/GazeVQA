"""Closed-form answer checks for questions whose reference is a yes/no, a count or a frame number.

No model is involved: the route is decided from the reference answers alone, and the model's free-text answer is searched
for its verdict (yes/no, a number, a frame number). Verbose answers state the verdict at the end, so the search goes
lead -> bold span -> explicit conclusion -> topic sentence, and every function returns (verdict, how) so that the
rule that fired can be audited. A verdict of None means "no verdict found": the caller falls back to the judge.

route(reference_answers)                  -> "yesno" | "number" | "frame" | "open"
gt_value(route, reference_answers)        -> "yes"/"no" | int | [(lo, hi), ...]
answer_polarity(answer, question)         -> ("yes" | "no" | None, how)
answer_count(answer, question)            -> (int | None, how)
answer_frame(answer)                      -> ((lo, hi) | None, how)
check(route, gt, answer, question)        -> dict(verdict, how, ok, frame_dist)
"""

import re

SEP = " ||| "

_WORDS = {"zero": 0, "no": 0, "none": 0, "one": 1, "a single": 1, "single": 1, "two": 2, "three": 3, "four": 4, "five": 5,
          "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "both": 2}
_NUMWORD = r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_COUNT_RE = re.compile(rf"\b(\d{{1,3}})\b|\b({_NUMWORD})\b", re.I)
_NOONE_RE = re.compile(r"\b(?:no\s+(?:one|body|people|person|children|child|individuals?)|nobody|none|zero)\b", re.I)
_FRAME_RE = re.compile(r"\bframes?\s*#?\s*(\d+)(?:\s*(?:-|–|—|to|through|and)\s*(?:frame\s*)?(\d+))?", re.I)
_YN_GT = re.compile(r"^\W*(yes|no)\b(?!\s+(?:one|body)\b)", re.I)
_GT_NONE = re.compile(r"^\s*there\s+(?:is|are)\s+no\b.*\bin the image", re.I)
_GT_FRAME = re.compile(r"^\s*frames?\s*(\d+)(?:\s*[-–—]\s*(\d+))?", re.I)
_GT_COUNT = re.compile(r"^\s*(\d+)\b")


def clean(text) -> str:
    """Markdown-light plain text: keeps the **bold** markers (they mark conclusions) but drops headings, bullets, rules."""
    t = str(text or "").replace("\r", "")
    t = re.sub(r"</?points?[^>]*>", "", t)  # Molmo2 pointing markup: the coordinates are not part of the answer text
    t = re.sub(r"^\s*#{1,6}\s*", "", t, flags=re.M)
    t = re.sub(r"^\s*[-*•]\s+", "", t, flags=re.M)
    t = re.sub(r"^\s*\d+[.)]\s+", "", t, flags=re.M)
    t = re.sub(r"[`_]{1,3}", "", t)
    t = re.sub(r"(\*\*(?:final answer|answer|conclusion|analysis)\*\*)\s*:?\s*\n+", r"\1: ", t, flags=re.I)
    return re.sub(r"[ \t]+", " ", t).strip()


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


def bold_spans(text: str) -> list[str]:
    return [m.strip() for m in re.findall(r"\*\*(.+?)\*\*", text, flags=re.S) if m.strip()]


# ---------------------------------------------------------------------------------------------- routing and references
def route(reference_answers) -> str:
    refs = [r.strip() for r in str(reference_answers or "").split(SEP) if r.strip()]
    if not refs:
        return "open"
    if all(_GT_FRAME.match(r) for r in refs):
        return "frame"
    pol = {m.group(1).lower() for r in refs if (m := _YN_GT.match(r))}
    if len(pol) == 1 and all(_YN_GT.match(r) for r in refs):
        return "yesno"
    vals = {_ref_count(r) for r in refs}
    if len(vals) == 1 and None not in vals:
        return "number"
    return "open"


def _ref_count(ref: str):
    if _GT_NONE.match(ref):
        return 0
    m = _GT_COUNT.match(ref)
    return int(m.group(1)) if m else None


def gt_value(rt: str, reference_answers):
    refs = [r.strip() for r in str(reference_answers or "").split(SEP) if r.strip()]
    if rt == "yesno":
        return _YN_GT.match(refs[0]).group(1).lower()
    if rt == "number":
        return _ref_count(refs[0])
    if rt == "frame":
        out = []
        for r in refs:
            m = _GT_FRAME.match(r)
            lo = int(m.group(1))
            out.append((lo, int(m.group(2)) if m.group(2) else lo))
        return out
    return None


# -------------------------------------------------------------------------------------------------------- yes / no
_NEG = re.compile(r"\b(?:there\s+(?:is|are)\s+(?:indeed\s+|clearly\s+)?no\b|there\s+(?:isn't|aren't|is\s+not|are\s+not)\b|no\s+(?:exchanged|shared|mutual|evidence|clear|direct|single|common|one)|"
                  r"nobody|(?:does|do|did)\s+not|(?:doesn't|don't|didn't)|(?:is|are)\s+not|(?:isn't|aren't)|cannot|can't|not\s+(?:any|exchang|shar|mutual|at each other|looking back)|"
                  r"never|neither|without)\b", re.I)
_POS = re.compile(r"\b(?:there\s+(?:is|are)\s+(?:indeed\s+|clearly\s+|definitely\s+)?(?:a|an|some|two|three|several|at least|exchanged|shared|mutual|instances?|evidence)\b|"
                  r"(?:instances?|examples?|cases?)\s+of\s+(?:exchanged|mutual|shared)|(?:do|does|are|is)\s+(?:exchang|shar|look(?:ing)?\s+at\s+each)|"
                  r"(?:exchanged|shared|mutual)\s+gaze\s+(?:target\s+)?(?:between|occurs|exists|is|are)\b|(?:shared\s+)?gaze\s+target\s+(?:in\s+the\s+image\s+)?is\b|"
                  r"(?:indicat|suggest|show|constitut|creat)\w*\s+(?:that\s+)?(?:they\s+are\s+)?(?:an?\s+)?(?:exchang|shar|mutual)|"
                  r"(?:common|shared)\s+(?:focal\s+point|point\s+of\s+focus|focus|target)|focal\s+point|"
                  r"(?:does|do)\s+look\s+(?:at\s+the\s+same\s+target\s+)?again|they\s+(?:are\s+)?(?:exchanging|looking\s+at\s+each)|mutual\s+gaze)", re.I)
_TOPIC = {"shared": ("shared", "same object", "same target", "same thing", "common", "same", "focal", "gaze target"),
          "exchanged": ("exchang", "mutual", "each other", "look back", "looking back", "reciprocat"),
          "again": ("again", "same target", "revisit", "look back", "returns", "return"),
          "block": ("block", "occlu", "cover", "obstruct", "hid")}
_RESTATE = re.compile(r"\b(?:to determine|to answer|to see|to check|to figure|whether|let's|let us|we need to|need to determine|"
                      r"if there|the question|asks? (?:if|whether))\b", re.I)


def _topic_words(question: str):
    q = str(question or "").lower()
    for key, words in _TOPIC.items():
        if key in q:
            return words
    return ()


def _group_polarity(span: str):
    """'Is there any person NOT engaged with the group?': Yes when somebody is left out, No when everybody is in."""
    s = span.lower()
    left_out = (r"(?:disengaged|distracted|left out|excluded|looking away|less\s+engaged|lack\s+of\s+engagement|"
                r"outside\s+(?:the\s+)?(?:immediate\s+)?group|not\s+(?:fully\s+|visually\s+|really\s+|directly\s+)?engaged|not\s+paying\s+attention)")
    if re.search(rf"\b(?:no\s+(?:one|body|person|child|individual)|nobody|none|(?:doesn't|does\s+not|don't|do\s+not|isn't|aren't|not)\s+(?:seem|appear)?\s*(?:to\s+be\s+)?(?:anyone|anybody|any\s+\w+))\b[^.]{{0,60}}\b{left_out}", s):
        return "no"
    if re.search(r"\b(?:every(?:one|body)|all(?:\s+(?:of\s+)?the\s+\w+|\s+\w+)?|each)\b[^.]{0,80}\b(?:engaged|focused|attending|participating|involved|paying attention)\b", s) \
            and not re.search(r"\bnot\s+(?:all|every)", s) and not re.search(rf"\b(?:every(?:one|body)|all)\b[^.]{{0,60}}\b{left_out}", s):
        return "no"
    if re.search(rf"\b(?:is|are|appears?|seems?|looks?)\s+(?:to\s+be\s+)?{left_out}|\b{left_out}", s):
        return "yes"
    return None


def _first_of(span: str):
    """Earliest of a negation / an assertion about the topic inside one sentence."""
    n, p = _NEG.search(span), _POS.search(span)
    if n and p:
        return "no" if n.start() < p.start() else "yes"
    return "no" if n else ("yes" if p else None)


def _polarity_of(span: str, topic, group=False):
    """Explicit yes / no word, else a negation or assertion of the question's topic inside the span."""
    if _RESTATE.search(span) and not re.search(r"\b(yes|no)\b", span, re.I):
        return None
    # a concession ("While there is some interaction, ...") is not the verdict: judge the main clause
    span = re.sub(r"^(?:\w+,\s*)?(?:while|although|though|even though|whereas)\b[^,]*,\s*", "", span, flags=re.I)
    if not span.strip():
        return None
    m = re.search(r"\b(yes|no)\b(?!\s+(?:one|body)\b)", span, re.I)
    if m:
        return m.group(1).lower()
    if group:
        return _group_polarity(span)
    if re.search(r"\bno\s+(?:one|body)\b|\bnobody\b", span, re.I):
        return "no"
    if topic and not any(w in span.lower() for w in topic):
        return None
    return _first_of(span)


_ANSWER_MARK = re.compile(r"(?:final\s+answer|answer|conclusion|in\s+conclusion|therefore|thus|so|hence)\W{0,14}\**\W{0,3}(yes|no)\b(?!\s+(?:one|body)\b)", re.I)


def answer_polarity(answer, question=""):
    t = clean(answer)
    if not t:
        return None, "empty"
    m = re.match(r"^\W*(yes|no)\b", t, re.I)
    if m:
        return m.group(1).lower(), "lead"
    topic = _topic_words(question)
    group = "not engaged" in str(question).lower()
    marks = list(_ANSWER_MARK.finditer(t))
    if marks:
        return marks[-1].group(1).lower(), "answer_marker"
    for span in reversed(bold_spans(t)):
        p = _polarity_of(span, topic, group)
        if p:
            return p, "bold"
    sents = sentences(t)
    markers = re.compile(r"\b(?:therefore|thus|so|in conclusion|overall|in summary|answer|conclusion|hence|to conclude|based on)\b", re.I)
    for s in reversed(sents[-5:]):
        if markers.search(s):
            p = _polarity_of(s, topic, group)
            if p:
                return p, "conclusion"
    for s in reversed(sents[-3:]):
        p = _polarity_of(s, topic, group)
        if p:
            return p, "last_sentences"
    for s in sents[:1]:
        p = _polarity_of(s, topic, group)
        if p:
            return p, "first_sentence"
    return None, "none"


# ----------------------------------------------------------------------------------------------------------- counts
def _count_in(span: str):
    """First count (digit or number word) in the span; "no one / nobody / none" = 0."""
    first = _COUNT_RE.search(span)
    noone = _NOONE_RE.search(span)
    cands = []
    if first:
        cands.append((first.start(), int(first.group(1)) if first.group(1) else _WORDS[first.group(2).lower()]))
    if noone:
        cands.append((noone.start(), 0))
    if not cands:
        return None
    return min(cands)[1]


_LOOK = re.compile(r"\b(?:look(?:s|ing)?|gaz(?:e|es|ing)|star(?:e|es|ing)|watch(?:es|ing)?|focus(?:es|ed|ing)?|direct(?:ed|s)?|point(?:s|ing)?)\b", re.I)
_PEOPLE = re.compile(r"\b(?:people|persons?|children|child|kids?|individuals?|men|man|women|woman|girls?|boys?|adults?|babies|baby|times?|objects?)\b", re.I)


def answer_count(answer, question=""):
    t = clean(answer)
    if not t:
        return None, "empty"
    m = re.match(r"^\W*(\d{1,3})\b(?!\s*(?:st|nd|rd|th)\b)", t) or re.match(rf"^\W*({_NUMWORD})\b", t, re.I)
    if m:
        s = m.group(1)
        return (int(s) if s.isdigit() else _WORDS[s.lower()]), "lead"
    if re.match(r"^\W*(?:no\s+(?:one|body)|nobody|none|there\s+(?:is|are)\s+no)\b", t, re.I):
        return 0, "lead_none"
    ends = list(re.finditer(r"\b(?:total|answer|count|final answer|in total|number)\s*(?:is|are|of [a-z ]+ is|of|=|:)?\s*\**\s*(\d{1,3}|" + _NUMWORD + r")\b", t, re.I))
    if ends:
        s = ends[-1].group(1)
        return (int(s) if s.isdigit() else _WORDS[s.lower()]), "answer_marker"
    sents = sentences(t)
    for span in reversed(bold_spans(t)):
        if _LOOK.search(span) or _NOONE_RE.search(span) or _PEOPLE.search(span):
            c = _count_in(span)
            if c is not None:
                return c, "bold"
    for s in reversed(sents[-3:]):
        if _LOOK.search(s) and _PEOPLE.search(s):
            c = _count_in(s)
            if c is not None:
                return c, "last_sentences"
    for s in sents[:2]:
        if _LOOK.search(s) or _NOONE_RE.search(s):
            c = _count_in(s)
            if c is not None:
                return c, "first_sentences"
    return None, "none"


# ------------------------------------------------------------------------------------------------------------ frames
def _frame_span(m):
    lo = int(m.group(1))
    hi = int(m.group(2)) if m.group(2) else lo
    return (min(lo, hi), max(lo, hi))


def answer_frame(answer):
    t = clean(answer)
    if not t:
        return None, "empty"
    for span in reversed(bold_spans(t)):
        m = _FRAME_RE.search(span)
        if m:
            return _frame_span(m), "bold"
    sents = sentences(t)
    m = re.match(r"^\W*(?:frames?\s*#?\s*)?(\d+)(?:\s*(?:-|–|—|to)\s*(\d+))?\W*$", t, re.I)
    if m:
        return _frame_span(m), "bare"
    for s in sents[:1]:
        m = _FRAME_RE.search(s)
        if m:
            return _frame_span(m), "first_sentence"
    for s in reversed(sents[-2:]):
        ms = list(_FRAME_RE.finditer(s))
        if ms:
            return _frame_span(ms[-1]), "last_sentences"
    # "appears in the video from the beginning" / "from 00:00": the video starts at frame 1
    if re.search(r"\b(?:from|at|since)\s+the\s+(?:very\s+)?(?:beginning|start)\b|\bfrom\s+(?:0:00|00:00|0\.0\s*(?:s|sec))", t, re.I):
        return (1, 1), "beginning"
    return None, "none"


# ------------------------------------------------------------------------------------------------------------ check
def check(rt: str, gt, answer, question="") -> dict:
    """verdict: what was found in the answer (None = nothing), ok: 1/0 vs the reference (None when no verdict)."""
    out = {"verdict": None, "how": "", "ok": None, "frame_dist": None}
    if rt == "yesno":
        v, how = answer_polarity(answer, question)
        out.update(verdict=v, how=how, ok=None if v is None else int(v == gt))
    elif rt == "number":
        v, how = answer_count(answer, question)
        out.update(verdict=v, how=how, ok=None if v is None else int(v == gt))
    elif rt == "frame":
        v, how = answer_frame(answer)
        if v is not None:
            dist = min(max(lo - v[1], v[0] - hi, 0) for lo, hi in gt)
            out.update(verdict=f"{v[0]}-{v[1]}" if v[0] != v[1] else str(v[0]), ok=int(dist == 0), frame_dist=dist)
        out["how"] = how
    return out
