from pathlib import Path
import os

from supermops.eval import PDEval

class EvalLoader:
    def __init__(self, root=None):
        if root is None:
            try:
                root = os.environ["SUPERMOPS_EVAL_ROOT"]
            except KeyError:
                root = ""
        if not root:
            root = "../../data/evaluation_results"
        self.eval_root = Path(root)
        self.folder_exact = self.eval_root / "exact"
        self.folder_noisy = self.eval_root / "noisy"

        self.params = [(0, 0), (3, 0), (5, 3), (10, 7)]
        self.new_dynsep = PDEval.from_hdf(self.folder_exact / "gt_adapted/redlin/eval/eval_all.h5")
        self.eval_dirs = dict()
        for mu, nu in self.params:
            self.eval_dirs[f"exact/fc2/t3gs100/mu{mu}nu{nu}"] = self.folder_exact / f"gt_t3/t3/mu{mu}nu{nu}"
            for t in [5, 7]:
                self.eval_dirs[f"exact/fc2/t{t}gs100/mu{mu}nu{nu}/old"] = self.folder_exact / f"gt_t3/t{t}/mu{mu}nu{nu}"
                self.eval_dirs[f"exact/fc2/t{t}gs100/mu{mu}nu{nu}/new"] = (self.new_dynsep, f"num_time_steps == {t} & num_mu_dirs == {mu} & num_extra_nu_dirs == {nu}")

        self.eval_dirs["exact/fc2/t3/adcg"] = self.folder_exact / "gt_t3/t3/adcg"
        self.new_dynsep_adcg = PDEval.from_hdf(self.folder_exact / "gt_adapted/adcg/eval/eval_all.h5")
        for t in [5, 7]:
            self.eval_dirs[f"exact/fc2/t{t}/adcg/old"] = self.folder_exact / f"gt_t3/t{t}/adcg"
            self.eval_dirs[f"exact/fc2/t{t}/adcg/new"] = (self.new_dynsep_adcg, f"num_time_steps == {t}")

        self.eval_dirs["noisy/wsteinvsdelta/adcg"] = self.folder_noisy / "adcg"
        self.eval_dirs["noisy/wsteinvsdelta/gs200/gtall"] = self.folder_noisy / "redlin"

        self.gs200_singledouble = PDEval.from_hdf(self.folder_noisy / "singledouble" / "eval/eval_all.h5")
        self.eval_dirs["noisy/wsteinvsdelta/gs200/gtsingle"] = (self.gs200_singledouble, "tag.str.contains('single')")
        self.eval_dirs["noisy/wsteinvsdelta/gs200/gtdouble"] = (self.gs200_singledouble, "tag.str.contains('double')")

    def load_eval(self, label):
        print(label)
        edir = self.eval_dirs[label]
        if isinstance(edir, tuple):
            return edir[0].filter(edir[1])
        else:
            return PDEval.from_hdf(self.eval_dirs[label] / "eval" / "eval_all.h5")

