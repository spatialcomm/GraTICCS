"""Original-tissue channels ordered by their best-ranked event (analysis/events.py).

python -m analysis.channel_rank contact G_real_s0,G_real_s1,G_real_s2 results/channel_best_rank_contact_G.csv
python -m analysis.channel_rank secreted G2_sec_real_s0,G2_sec_real_s1,G2_sec_real_s2 results/channel_best_rank_secreted_G2.csv
"""
import json
import sys

import numpy as np
import pandas as pd

from analysis.events import D, scores
from graticcs.paths import RESULTS

kind, runs, out = sys.argv[1], [RESULTS / x for x in sys.argv[2].split(",")], sys.argv[3]
s = scores(kind, runs); ch, real = s["ch"], s["real"]
lit = json.load(open(D / "lit_channel_matches.json"))[kind]
best = s["rank"].min(0)
d = pd.DataFrame(dict(channel=ch.name[real].to_numpy(), stratum=ch.stratum[real].to_numpy(), p=s["p"][real], q=s["q"][real],
                      best_readout=[s["T"][i] for i in s["bt"][real]], best_rank=best, n_events=s["score"].size,
                      lit=[ch.name[c] in lit for c in real])).sort_values("best_rank")
d["channel_rank_by_best_event"] = np.arange(1, len(d) + 1)
d.to_csv(out, index=False)
print(d.head(30).to_string(index=False))
