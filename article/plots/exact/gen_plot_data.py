import json

import numpy as np
import pandas as pd

from loader import EvalLoader

def argwhere_correct(correct, stat, dyn):
    return set(np.argwhere((correct[0] == stat) & (correct[1] == dyn)).flatten() + 1)

params = ["mu0nu0", "mu3nu0", "mu5nu3", "mu10nu7", "adcg"]
loader = EvalLoader()

res = {"blobs": dict(), "lines": dict()}
for t in [3, 5, 7]:
    if t == 3:
        evals = {p: loader.load_eval(f"exact/fc2/t{t}gs100/{p}") for p in params[:-1]}
        evals["adcg"] = loader.load_eval(f"exact/fc2/t{t}/adcg")
        blob_lbls = ["mu0nu0", "mu5nu3"]
        line_lbls = params
    else:
        evals = {f"{p}/old": loader.load_eval(f"exact/fc2/t{t}gs100/{p}/old") for p in params[:-1]}
        evals.update({f"{p}/new": loader.load_eval(f"exact/fc2/t{t}gs100/{p}/new") for p in params[:-1]})
        evals["adcg"] = loader.load_eval(f"exact/fc2/t{t}/adcg/new")
        blob_lbls = ["mu0nu0/old", "mu5nu3/old"]
        line_lbls = [p + "/new" for p in params[:-1]] + ["adcg"]

    # blobs
    correct = np.array([evals[lbl].cluster_correct()["cluster_correct"].tolist() for lbl in blob_lbls])
    matrix = np.array([[argwhere_correct(correct, 1-i, 1-j) for j in range(2)] for i in range(2)])
    sum_mat = np.array([[len(el) for el in row] for row in matrix])
    frame = pd.DataFrame(sum_mat, index=["stat_good", "stat_bad"], columns=["dyn_good", "dyn_bad"])
    print(frame)
    print(frame.to_dict())
    res["blobs"][t] = frame.to_dict()
    # lines
    res["lines"][t] = dict()
    bins = None
    for parm, lbl in zip(params, line_lbls):
        eres_correct = evals[lbl].cluster_correct()
        if bins is None:
            binned, bins = pd.cut(eres_correct["dyn_sep"], 7, retbins=True)
            res["lines"][t]["bins"] = bins.tolist()
        else:
            binned = pd.cut(eres_correct["dyn_sep"], bins)
        res["lines"][t][parm] = eres_correct["cluster_correct"].groupby(binned).mean().tolist()

with open("plot_data.json", "w") as file:
    json.dump(res, file, indent=4)
