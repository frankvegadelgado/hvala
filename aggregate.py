"""
aggregate.py --- per-family aggregation of ablation JSON records.

Reads one or more JSON files produced by `ablation.py`, groups the records
by benchmark family (FRB, brock200, brock400, ..., bio, ca, scc, socfb,
crown, odd-cycle, named-cubic, bipartite-expander), and computes, per
family:

  * the number of instances `n`,
  * the mean cover size of each candidate `c1`, `c2`, `c3`, `c4`,
  * the "% min-attaining" rate of each candidate
    (fraction of instances on which |c_i| == min(c1,c2,c3,c4)).

Emits a single summary JSON to stdout. Output format:

    {
      "per_family": {
        "FRB": {"n": 41, "mean": {"c1":1053.0,...}, "min_pct": {"c1":32,...}},
        ...
      },
      "overall": {"n": 252, "mean": {...}, "min_pct": {...}}
    }

Usage:

    python aggregate.py frb.json dimacs.json realworld.json hardest.json \
        > ablation_summary.json

The summary JSON is the direct input to `make_tables.py`.
"""

import sys
import json
from collections import defaultdict


# ---------- family taxonomy ----------


def family_of(name):
    """Map an instance name to its benchmark family."""
    n = name.replace(".mis", "").replace(".clq", "")
    n = n.replace(".clq-compliment.txt", "")
    n = n.replace(".txt", "")
    lower = n.lower()

    # FRB
    if lower.startswith("frb"):
        return "FRB"

    # NPBench DIMACS clique-complement prefixes
    for pref in (
        "brock800", "brock400", "brock200",
        "c-fat500", "c-fat200",
        "hamming10", "hamming8", "hamming6",
        "johnson32", "johnson16", "johnson8",
        "MANN_a81", "MANN_a45", "MANN_a27", "MANN_a9",
        "keller5", "keller4",
        "p_hat1500", "p_hat1000", "p_hat700", "p_hat500", "p_hat300",
        "san1000", "san400", "san200", "sanr400", "sanr200",
        "gen400", "gen200",
        "C1000", "C500", "C250", "C125",
    ):
        if n.startswith(pref):
            return pref

    # Adversarial families (experiment/hardest/)
    if n.startswith("crown-"):
        return "Crown"
    if n.startswith("odd-cycle-"):
        return "Odd cycle"
    if n in {"petersen", "heawood", "mobius-kantor", "pappus", "desargues"}:
        return "Named cubic"
    if n.startswith("bip-3reg-"):
        return "Bipartite expander"

    # Real-world prefixes
    if lower.startswith("bio-"):
        return "Bio"
    if lower.startswith("ca-"):
        return "Collab"
    if lower.startswith("ia-email") or lower.startswith("ia-enron"):
        return "Email"
    if lower.startswith("ia-fb") or lower.startswith("soc-"):
        return "Social"
    if lower.startswith("ia-infect") or lower.startswith("ia-reality"):
        return "Infect/Reality"
    if lower.startswith("ia-wiki") or lower.startswith("web-wikipedia"):
        return "Wiki"
    if lower.startswith("inf-"):
        return "Infra"
    if lower.startswith("rec-"):
        return "Rec"
    if lower.startswith("rt-"):
        return "Retweet"
    if lower.startswith("sc-"):
        return "SciComp"
    if lower.startswith("scc"):
        return "SCC"
    if lower.startswith("socfb-"):
        return "SocFB"
    if lower.startswith("tech-"):
        return "Tech"
    if lower.startswith("web-"):
        return "Web"

    return "Other"


# ---------- summary helpers ----------


def summarise(records):
    n = len(records)
    if n == 0:
        return {"n": 0,
                "mean": {"c1": 0, "c2": 0, "c3": 0, "c4": 0},
                "min_pct": {"c1": 0, "c2": 0, "c3": 0, "c4": 0}}
    mean = {k: sum(r[k] for r in records) / n for k in ("c1", "c2", "c3", "c4")}
    ties = {k: 0 for k in ("c1", "c2", "c3", "c4")}
    for r in records:
        m = min(r["c1"], r["c2"], r["c3"], r["c4"])
        for k in ("c1", "c2", "c3", "c4"):
            if r[k] == m:
                ties[k] += 1
    pct = {k: round(100.0 * ties[k] / n, 1) for k in ties}
    return {"n": n, "mean": {k: round(mean[k], 1) for k in mean}, "min_pct": pct}


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)

    by_family = defaultdict(list)
    all_records = []
    for path in sys.argv[1:]:
        with open(path) as f:
            data = json.load(f)
        for r in data.get("results", []):
            if "c1" not in r:
                continue
            by_family[family_of(r["name"])].append(r)
            all_records.append(r)

    per_family = {fam: summarise(rs) for fam, rs in by_family.items()}
    overall = summarise(all_records)
    json.dump({"per_family": per_family, "overall": overall},
              sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
