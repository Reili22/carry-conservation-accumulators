"""Minimal Liberty reader: pin directions, functions, capacitances, ff groups, energies.

Energy model (zero-delay, documented in the paper): an output toggle costs the
mean of its rise/fall internal-energy tables (averaged over the table) plus
0.5*C*V^2 of the driven pin capacitance; input pins with hidden power cost the
mean of their tables per toggle.  Liberty energy unit = V^2*fF = fJ.
"""
import re
from functools import lru_cache

_TOK = re.compile(r'"(?:[^"\\]|\\.)*"|[{}();:,]|[^\s{}();:,"]+')


def _tokens(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S).replace("\\\n", " ")
    return _TOK.findall(text)


def _parse(tokens, i):
    """Parse group body starting after '{'; return (dict(attrs, groups), next index)."""
    node = {"attrs": {}, "groups": []}
    while i < len(tokens):
        t = tokens[i]
        if t == "}":
            return node, i + 1
        name = t
        if tokens[i + 1] == ":":
            val = tokens[i + 2]
            node["attrs"][name] = val.strip('"')
            i += 3
            if i < len(tokens) and tokens[i] == ";":
                i += 1
            continue
        if tokens[i + 1] == "(":
            j = i + 2
            args = []
            while tokens[j] != ")":
                if tokens[j] != ",":
                    args.append(tokens[j].strip('"'))
                j += 1
            j += 1
            if j < len(tokens) and tokens[j] == "{":
                child, j = _parse(tokens, j + 1)
                child["type"], child["args"] = name, args
                node["groups"].append(child)
            else:
                node["attrs"][name] = args
                if j < len(tokens) and tokens[j] == ";":
                    j += 1
            i = j
            continue
        i += 1
    return node, i


def _floats(v):
    s = v[0] if isinstance(v, list) else v
    return [float(x) for x in re.split(r"[,\s]+", " ".join(v) if isinstance(v, list) else s) if x]


def _table_mean(group):
    vals = group["attrs"].get("values")
    return sum(_floats(vals)) / len(_floats(vals)) if vals else 0.0


@lru_cache(maxsize=2)
def read(path):
    text = open(path).read()
    toks = _tokens(text)
    lib, _ = _parse(toks, toks.index("{") + 1)
    volt = float(lib["attrs"].get("nom_voltage", 1.0))
    cells = {}
    for g in lib["groups"]:
        if g.get("type") != "cell":
            continue
        name = g["args"][0]
        cell = {"area": float(g["attrs"].get("area", 0)), "leak_nW": float(g["attrs"].get("cell_leakage_power", 0)),
                "pins": {}, "ff": None}
        for p in g["groups"]:
            if p["type"] == "ff":
                cell["ff"] = {"q": p["args"][0], "qn": p["args"][1], "next": p["attrs"]["next_state"],
                              "clk": p["attrs"]["clocked_on"]}
            if p["type"] != "pin":
                continue
            pin = {"dir": p["attrs"].get("direction"), "cap": float(p["attrs"].get("capacitance", 0)),
                   "func": p["attrs"].get("function"), "e_int": 0.0}
            ints = [ip for ip in p["groups"] if ip["type"] == "internal_power"]
            if ints:
                per = []
                for ip in ints:
                    rf = [_table_mean(t) for t in ip["groups"] if t["type"] in ("rise_power", "fall_power")]
                    if rf:
                        per.append(sum(rf) / len(rf))
                pin["e_int"] = sum(per) / len(per) if per else 0.0
            cell["pins"][p["args"][0]] = pin
        cells[name] = cell
    return {"V": volt, "cells": cells}


def py_function(expr):
    """Convert a Liberty boolean function to a Python expression over numpy bool arrays."""
    e = expr.replace("!", "~").replace("+", "|").replace("*", "&")
    e = re.sub(r"(\w)\s+(?=[\w(~])", r"\1 & ", e)  # implicit AND by juxtaposition
    return e
