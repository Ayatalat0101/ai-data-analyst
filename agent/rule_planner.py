"""
Rule-based planner: question -> Plan, with no LLM.

Why keep a non-AI planner?
--------------------------
1. It is the MVP: the whole pipeline (validate -> tool -> answer) can be built
   and tested before any API call exists.
2. It is the fallback when there is no key, no internet or the daily quota ends.
3. It is a baseline: later we can measure how much the LLM actually improves.

It knows nothing about aid distributions. Everything comes from the profile
(column names, types and category values), so it works on any CSV.
"""
from __future__ import annotations

import re

from .data_loader import DatasetProfile
from .plan import FilterSpec, Plan, clarify, reject

UNSAFE = re.compile(r"\b(delete|drop|remove|insert|update|overwrite|modify|rename|"
                    r"import|exec|eval|subprocess|os\.|sys\.|__\w+__|rm\s+-)\b", re.I)
OUT_OF_SCOPE = re.compile(r"\b(predict|forecast|projection|next\s+(week|month|quarter|year)|"
                          r"will\s+(we|it|they)|in\s+the\s+future|machine\s+learning|train\s+a\s+model|"
                          r"correlat\w*|regression|why\s+did)\b", re.I)
VAGUE = re.compile(r"\b(best|worst|good|bad|big|bigger|biggest|small|smaller|smallest|"
                   r"large|larger|largest|important|performing|top)\b", re.I)

AGG_WORDS = [  # order matters: first match wins
    (r"\b(average|mean|avg)\b", "mean"),
    (r"\b(total|sum|overall)\b", "sum"),
    (r"\b(maximum|max|highest value)\b", "max"),
    (r"\b(minimum|min|lowest value)\b", "min"),
]
MOST = re.compile(r"\b(most|highest|largest|biggest|greatest|maximum|top)\b", re.I)
LEAST = re.compile(r"\b(least|lowest|smallest|fewest|minimum|bottom)\b", re.I)
COUNT = re.compile(r"\b(how\s+many|count|number\s+of)\b", re.I)
LIST = re.compile(r"^\s*(show|list|display|give|find|get|which|what\s+are)\b", re.I)
GROUP = re.compile(r"\b(by|per|for\s+each|each|across|breakdown\s+of)\s+", re.I)
TIME = re.compile(r"\b(monthly|per\s+month|by\s+month|each\s+month|weekly|per\s+week|by\s+week|"
                  r"yearly|per\s+year|by\s+year|daily|per\s+day|by\s+day)\b", re.I)
TOP_N = re.compile(r"\b(?:top|bottom|first|largest|highest|lowest|smallest)\s+(\d{1,2})\b", re.I)
COMPARE = re.compile(
    r"(?P<op>greater\s+than\s+or\s+equal\s+to|less\s+than\s+or\s+equal\s+to|at\s+least|at\s+most|"
    r"greater\s+than|more\s+than|less\s+than|fewer\s+than|above|over|below|under|"
    r"equal\s+to|equals|>=|<=|>|<|=)\s*(?P<num>-?\d[\d,]*(?:\.\d+)?)", re.I)
OP_MAP = {"greater than or equal to": ">=", "at least": ">=", "less than or equal to": "<=",
          "at most": "<=", "greater than": ">", "more than": ">", "above": ">", "over": ">",
          "less than": "<", "fewer than": "<", "below": "<", "under": "<",
          "equal to": "==", "equals": "==", "=": "==", ">=": ">=", "<=": "<=", ">": ">", "<": "<"}
AGG_TARGET = re.compile(r"\b(?:average|mean|avg|total|sum|maximum|minimum|max|min)\s+(?:of\s+)?(?:the\s+)?"
                        r"(?P<target>[a-z][a-z\s]*?)(?=\s+(?:per|by|for|in|of|across|where|with|and|"
                        r"from|at|on|was|were|is|are)\b|[?.!,]|$)", re.I)
STOP = {"the", "a", "an", "all", "each", "every", "row", "rows", "record", "records", "it", "them",
        "value", "values", "data", "number", "amount"}


# ----------------------------------------------------------------------------
# Vocabulary built from the profile
# ----------------------------------------------------------------------------
def _singular(w: str) -> str:
    return w[:-1] if w.endswith("s") and len(w) > 3 else w


def _aliases(profile: DatasetProfile) -> dict[str, str]:
    """Phrase -> column. 'households_reached' is reachable as 'households reached',
    'households' or 'household'. Tokens shared by two columns are dropped."""
    alias, token_owner = {}, {}
    for col in profile.column_names:
        spaced = col.replace("_", " ").casefold()
        alias[spaced] = col
        alias[_singular(spaced)] = col
        for tok in spaced.split():
            if len(tok) < 4 or tok in STOP:
                continue
            for t in {tok, _singular(tok)}:
                token_owner.setdefault(t, set()).add(col)
    for tok, owners in token_owner.items():
        if len(owners) == 1 and tok not in alias:
            alias[tok] = next(iter(owners))
    return alias


def _find_columns(text: str, aliases: dict[str, str]) -> list[tuple[int, str]]:
    """(position, column) for each column mentioned, longest phrase first."""
    found, taken = [], []
    for phrase in sorted(aliases, key=len, reverse=True):
        for m in re.finditer(rf"\b{re.escape(phrase)}s?\b", text):
            if any(a <= m.start() < b for a, b in taken):
                continue
            taken.append((m.start(), m.end()))
            found.append((m.start(), aliases[phrase]))
    return sorted(found)


def _find_values(text: str, profile: DatasetProfile) -> list[FilterSpec]:
    """Category values written in the question become filters ('completed' -> status == Completed)."""
    by_col: dict[str, list[str]] = {}
    for col in profile.columns:
        for v in col.values or []:
            if re.search(rf"\b{re.escape(v.casefold())}(?:s|es)?\b", text):   # "kits" -> "Kit"
                by_col.setdefault(col.name, []).append(v)
    return [FilterSpec(column=c, op="==", value=v[0]) if len(v) == 1
            else FilterSpec(column=c, op="in", value=v) for c, v in by_col.items()]


def _near_miss_values(text: str, profile: DatasetProfile, aliases: dict[str, str],
                      exact: list[FilterSpec]) -> list[FilterSpec]:
    """'Khan Yunis' is not a value, but it is close to 'Khan Younis'.
    We return it as a filter with the user's spelling on purpose: the validator
    will see the unknown value and ASK 'Did you mean Khan Younis?'.
    Dropping it silently would answer a different question (all rows)."""
    from difflib import SequenceMatcher
    known = {a for a in aliases} | STOP | {"distribution", "distributions", "total", "average",
                                          "show", "which", "many", "count", "number"}
    exact_cols = {f.column for f in exact}
    words = re.findall(r"[a-z][a-z'\-]+", text)
    grams = {" ".join(words[i:i + k]) for k in (1, 2, 3) for i in range(len(words) - k + 1)}
    grams = {g for g in grams if len(g) >= 4 and g not in known}

    out = []
    for col in profile.columns:
        if not col.values or col.name in exact_cols:
            continue
        best = max(((SequenceMatcher(None, g, v.casefold()).ratio(), g, v)
                    for g in grams for v in col.values), default=(0, "", ""))
        ratio, gram, value = best
        if ratio >= 0.8 and gram != value.casefold():
            original = re.search(re.escape(gram), text).group(0)
            out.append(FilterSpec(column=col.name, op="==", value=original.title()))
    return out


def _numeric_filters(text: str, cols: list[tuple[int, str]], numeric: list[str]):
    """'quantity greater than 600' -> quantity > 600. Returns filters + columns consumed."""
    filters, used, pending = [], set(), []
    for m in COMPARE.finditer(text):
        before = [c for pos, c in cols if pos < m.start() and c in numeric]
        if not before:
            pending.append(m.group(0))
            continue
        col = before[-1]
        used.add(col)
        filters.append(FilterSpec(column=col, op=OP_MAP[re.sub(r"\s+", " ", m.group("op").lower())],
                                  value=float(m.group("num").replace(",", ""))))
    return filters, used, pending


RATE_WORDS = re.compile(r"(unit|price|rate|ratio|percent|pct|avg|average|mean|per_|_per|score|age)", re.I)


def is_rate_column(name: str) -> bool:
    """A price or rate per unit is averaged, never summed:
    'total unit cost' of 60 distributions is a meaningless number."""
    return bool(RATE_WORDS.search(name))


def _rank_options(group_by: str, numeric: list[str]) -> list[str]:
    """Complete, unambiguous questions the user can click."""
    opts = [f"Which {group_by} has the highest {'average' if is_rate_column(c) else 'total'} "
            f"{c.replace('_', ' ')}?" for c in numeric[:3]]
    return (opts + [f"Which {group_by} has the most rows?"])[:4]


# ----------------------------------------------------------------------------
# Planner
# ----------------------------------------------------------------------------
def plan_question(question: str, profile: DatasetProfile) -> Plan:
    q = question.strip()
    # "households_reached" and "households reached" must mean the same thing
    text = " " + q.casefold().replace("_", " ") + " "

    # 1. safety and scope come first ----------------------------------------
    if UNSAFE.search(text):
        return reject("I can only read and analyse this file. I can't change data or run code.",
                      "unsafe")
    if OUT_OF_SCOPE.search(text):
        return reject("Forecasting and explaining causes are outside my scope. I can summarise what "
                      "is already in the data, for example totals per month so far.", "out_of_scope",
                      reason="Question asks for prediction or causation.")

    aliases = _aliases(profile)
    numeric = profile.columns_of_kind("number")
    groupable = profile.columns_of_kind("category", "date")
    date_cols = profile.columns_of_kind("date")

    cols = _find_columns(text, aliases)
    value_filters = _find_values(text, profile)
    value_filters += _near_miss_values(text, profile, aliases, value_filters)
    num_filters, used_in_filters, pending = _numeric_filters(text, cols, numeric)
    if pending:
        return clarify(f"Which column should be '{pending[0]}'?",
                       [f"{q} ({c})" for c in numeric[:3]])
    filters = value_filters + num_filters

    # 2. what is being measured and how it is grouped -------------------------
    metrics = [c for _, c in cols if c in numeric and c not in used_in_filters]
    metric = metrics[0] if metrics else None

    group_by, time_grain = None, None
    tm = TIME.search(text)
    if tm and date_cols:
        group_by = date_cols[0]
        time_grain = next(g for g in ("month", "week", "year", "day") if g in tm.group(0) or
                          (g == "day" and "daily" in tm.group(0)) or (g == "year" and "yearly" in tm.group(0)))
    else:
        g = GROUP.search(text)
        after = [c for pos, c in cols if g and pos >= g.end() and c in groupable]
        which = re.search(r"\bwhich\s+(\w+(?:\s\w+)?)", text)
        which_col = [c for pos, c in cols if which and which.start() <= pos <= which.end() and c in groupable]
        if after:
            group_by = after[0]
        elif which_col:
            group_by = which_col[0]
    # "which is the best partner?" -> the subject is the grouping column mentioned
    if not group_by and (VAGUE.search(text) or MOST.search(text) or LEAST.search(text)):
        mentioned = [c for _, c in cols if c in groupable]
        group_by = mentioned[0] if mentioned else None
    agg = next((a for pat, a in AGG_WORDS if re.search(pat, text)), None)

    # 3. missing column: "average beneficiary age" ---------------------------
    t = AGG_TARGET.search(text)
    if t and not metric:
        target = t.group("target").strip()
        target_words = [w for w in target.split() if w not in STOP]
        mentions = _find_columns(" " + target + " ", aliases)
        if target_words and not mentions:
            return reject(f"I can't find a column for '{target}' in this file. Numeric columns I can "
                          f"calculate: {', '.join(numeric)}.", "missing_column")

    # 4. vague words without a measurable meaning -> ask ---------------------
    vague = VAGUE.search(text)
    counts_rows = bool(re.search(r"\b(distributions?|rows?|records?|entries|times|count)\b", text)) \
        and not metric and bool(MOST.search(text) or LEAST.search(text))
    if vague and not metric and not counts_rows and not TOP_N.search(text):
        subject = f"{group_by}" if group_by else "rows"
        word = vague.group(0)
        options = _rank_options(group_by, numeric) if group_by else \
                  [f"Show the top 5 rows by {c.replace('_', ' ')}" for c in numeric[:3]]
        return clarify(f"'{word.capitalize()}' can be measured in different ways. "
                       f"How should I compare {subject}?", options[:4],
                       reason=f"'{word}' has no measurable definition in the data.")

    # 5. choose the action ----------------------------------------------------
    n_match = TOP_N.search(text)
    wants_rank = bool(MOST.search(text) or LEAST.search(text) or n_match)
    ascending = bool(LEAST.search(text))

    if group_by and wants_rank and not time_grain and not metric and not counts_rows:
        # "which partner is highest?" - highest in WHAT? Never default to counting rows.
        return clarify(f"Highest by which measure? Choose how to rank each {group_by}:",
                       _rank_options(group_by, numeric),
                       reason="Ranking requested without a measurable column.")

    if group_by and wants_rank and not time_grain:
        n =int(n_match.group(1)) if n_match else (1 if re.search(r"\bwhich\b", text) else 5)
        return Plan(action="top_n", group_by=group_by, column=metric, agg="count" if not metric else (agg or "sum"),
                    n=n, ascending=ascending, filters=filters,
                    reason=f"Ranks each {group_by} by {'row count' if not metric else metric}.")

    if n_match and metric and not group_by:
        return Plan(action="top_n", column=metric, n=int(n_match.group(1)), ascending=ascending,
                    filters=filters, reason=f"Ranks individual rows by {metric}.")

    if agg and metric:
        if group_by:
            return Plan(action="group_aggregate", column=metric, agg=agg, group_by=group_by,
                        time_grain=time_grain, filters=filters,
                        reason=f"Calculates the {agg} of {metric} for each {group_by}.")
        return Plan(action="aggregate", column=metric, agg=agg, filters=filters,
                    reason=f"Calculates the {agg} of {metric}.")

    if COUNT.search(text):
        if group_by:
            return Plan(action="group_aggregate", agg="count", group_by=group_by, time_grain=time_grain,
                        filters=filters, reason=f"Counts rows for each {group_by}.")
        return Plan(action="count_rows", filters=filters, reason="Counts rows matching the filters.")

    if group_by and (metric or time_grain):
        return Plan(action="group_aggregate", column=metric, agg=agg or ("sum" if metric else "count"),
                    group_by=group_by, time_grain=time_grain, filters=filters,
                    reason=f"Summarises {metric or 'rows'} per {group_by}.")

    if LIST.search(text) or filters:
        return Plan(action="filter_rows", filters=filters, sort_by=metric,
                    ascending=ascending, reason="Lists the rows that match the filters.")

    cats = profile.columns_of_kind("category") or groupable
    examples = [f"How many rows are there?",
                f"Total {numeric[0]} by {cats[0]}" if numeric and cats else None,
                f"Average {numeric[0]}" if numeric else None]
    return clarify("I'm not sure what to calculate. Try one of these:",
                   [e for e in examples if e], reason="No known intent or column found.")
