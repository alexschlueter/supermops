import json

from loader import EvalLoader

def wsteinmean_per_noiselvl(eres, head=None):
    merged = eres.run[["noise_lvl"]].merge(eres.wstein.query("time == 0.0 & radius == 0.05")[["pjob_id", "sqwstein"]], on="pjob_id")
    assert merged["pjob_id"].is_unique
    group = merged.groupby("noise_lvl", as_index=True)["sqwstein"]
    #print(list(group))
    if head is None:
        return group.mean()
    else:
        return group.apply(lambda x: x.head(head)).mean(level=0)

linetags = {
    "mu5nu3_single": "noisy/wsteinvsdelta/gs200/gtsingle",
    "mu5nu3_double": "noisy/wsteinvsdelta/gs200/gtdouble",
    "mu5nu3_all": "noisy/wsteinvsdelta/gs200/gtall",
    "adcg": "noisy/wsteinvsdelta/adcg",
}

res = {"lines": {}}
loader = EvalLoader()
noise_lvls = None
for line, tag in linetags.items():
    eval_res = loader.load_eval(tag)
    line_data = wsteinmean_per_noiselvl(eval_res)
    if noise_lvls is None:
        noise_lvls = line_data.index.tolist()
    else:
        assert noise_lvls == line_data.index.tolist()
    # print(noise_lvls)
    # print(line_data)
    res["lines"][line] = line_data.tolist()
res["noise_lvls"] = noise_lvls

with open("plot_data.json", "w") as file:
    json.dump(res, file, indent=4)