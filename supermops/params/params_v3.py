import argparse
import json
import logging
from collections.abc import Mapping
from pathlib import Path

from supermops.utils.utils import parse_int_set, int_to_str_ranges


class Parameter:
    """A parameter in an experiment."""

    def __init__(self, name, fixed=False, default=None, type=None, perf=False, required=True, descr=None):
        """
        Args:
            name: Name
            fixed: True if the parameter always has the same fixed value, given
                   as default argument. Defaults to False.
            default: Default value.
            type: Function to convert a string representation of a value to a
                  specific Python type, e.g. int, bool, float, str
            perf: Does this parameter significantly change the resource / time
                  usage while running the experiment? Used for allocation of
                  resources on compute cluster. Defaults to False.
            required: Defaults to True.
            descr: Short text description.
        """
        self.name = name
        self.fixed = fixed
        if fixed:
            assert default is not None
        self.default = default
        self.type = type
        self.perf = perf
        self.required = required
        self.descr = descr

    def option_name(self):
        return "--" + self.name.replace("_", "-")

    def add_to_argparser(self, parser):
        # parser.add_argument(self.option_name(), default=self.default)
        if self.type == bool:
            parser.add_argument(self.option_name(), action=argparse.BooleanOptionalAction, help=self.descr)
        else:
            parser.add_argument(self.option_name(), type=self.type, help=self.descr)

class EMreconParmfileParameter(Parameter):
    def format_emrecon(self, value):
        return f"{self.name.upper()}={value}"

# Global dictionaries of possible parameters for different methods, each mapping
# a string name to an instance of Parameter:

# Parameters for the Python implementation of the dimension-reduced problem
redlin_params = {}
# Parameters for the Julia implementation of the ADCG algorithm applied to the
# full-dimensional dynamic problem
adcg_params = {}
# Parameters for the C implementation of the Primal-Dual algorithm applied to
# the dimension-reduced problem in the EMrecon tomography reconstruction
# software
emrecon_params = {}

def add_param(name, *args, **kwargs):
    redlin_params[name] = Parameter(name, *args, **kwargs)

def add_adcg_param(name, *args, **kwargs):
    adcg_params[name] = Parameter(name, *args, **kwargs)

def add_emrecon_param(name, *args, **kwargs):
    emrecon_params[name] = Parameter(name, *args, **kwargs)

# class MyJSONEncoder(json.JSONENCODER):
#     def default(self, obj):
#         try:
#             return obj.to_json()
#         except AttributeError:
#             return obj.

def pjobs_to_file(pjobs, file):
    json.dump(pjobs, file, indent=4, default=lambda o: o.to_json())

class Settings(Mapping):
    """A set of parameters and their values."""

    def __init__(self, pdict, **kwargs):
        self.settings = dict()
        self.update_dict(pdict, **kwargs)
        self.set_defaults()
        self.check_required()

    def update_dict(self, pdict, accept_none=False, error_unknown=True, error_fixed=True):
        """Update the internal dictionary, optionally checking for consistency.

        Args:
            pdict: Dictionary of new parameter name (str)->value pairs
            accept_none: Whether to accept a value of None. Defaults to False.
            error_unknown: Whether to raise an exception when a parameter with
                           unknown name is encountered. Defaults to True.
            error_fixed: Whether to raise an exception when a fixed parameter is
                         updated. Defaults to True.
        """
        for name, value in pdict.items():
            if accept_none or value is not None:
                try:
                    param = self.params[name]
                    if param.fixed:
                        if error_fixed:
                            raise Exception(f'Param "{name}" is fixed')
                    else:
                        self.settings[name] = param.type(value)
                except KeyError:
                    if error_unknown:
                        raise Exception(f'Param "{name}" unknown')

    def set_defaults(self):
        for name, param in self.params.items():
            if name not in self.settings and param.default is not None:
                self.settings[name] = param.default

    def check_required(self):
        for name, param in self.params.items():
            if param.required and name not in self.settings:
                raise Exception(f'Param "{name}" required but missing and no default given')

    @classmethod
    def from_cmdargs(cls, args):
        return cls(vars(args), accept_none=False, error_unknown=False)

    @classmethod
    def add_params_to_parser(cls, parser):
        group = parser.add_argument_group(cls.__name__)
        for param in cls.params.values():
            param.add_to_argparser(group)

    def update_from_cmdargs(self, args):
        self.update_dict(vars(args), accept_none=False, error_unknown=False)

    def __repr__(self):
        return type(self).__name__ + self.settings.__repr__()

    def to_json(self):
        return {
            "type": type(self).__name__,
            **self.settings
        }

    def __getitem__(self, idx):
        return self.settings[idx]

    def __iter__(self):
        return self.settings.__iter__()

    def __len__(self):
        return self.settings.__len__()

class RedlinSettings(Settings):
    params = redlin_params

class ADCGSettings(Settings):
    params = adcg_params

class EMreconSettings(Settings):
    params = emrecon_params

def add_pjobfilter_args(parser):
    group = parser.add_argument_group("Job filter")
    group.add_argument("--pjobs")
    group.add_argument("--filter", type=str)
    group.add_argument("--only-missing", action="store_true")
    group.add_argument("--only-eval-missing", action="store_true")

def add_pjob_args(parser, job=True):
    group = parser.add_argument_group("Job control")
    if job:
        group.add_argument("--jobid", type=int, default=1)
        group.add_argument("--reduction", type=int)
    group.add_argument("--param-file")
    group.add_argument("--param-json")
    add_pjobfilter_args(parser)

def pjob_json_decoder(dct):
    typ = dct.pop("type", None)
    if typ == "RedlinSettings":
        return RedlinSettings(dct, error_fixed=False)
    elif typ == "ADCGSettings":
        return ADCGSettings(dct, error_fixed=False)
    elif typ == "EMreconSettings":
        return EMreconSettings(dct, error_fixed=False)
    else:
        return dct

def filter_pjobs(pjobs, args):
    if args.pjobs is None:
        pjob_ids = list(range(1, len(pjobs) + 1))
    else:
        pjob_ids = sorted(parse_int_set(args.pjobs))

    if args.only_missing:
        # TODO: handle different recon res file params (currently difference between python + C)
        pjob_ids = [i for i in pjob_ids if not Path(pjobs[i-1]["res_file"]).exists()]
        # pjob_ids = [i for i in pjob_ids if not Path(pjobs[i-1]["recon_stats_file"]).exists()]

    if args.only_eval_missing:
        pjob_ids = [i for i in pjob_ids if not Path(f"eval/eval_{i}.pickle").exists()]

    if args.filter is not None:
        logging.info(f"Pjobs before filter: {len(pjob_ids)}")
        # pjobs = list(filter(lambda pjob: eval(args.filter, {"p": pjob}), pjobs))
        pjob_ids = [i for i in pjob_ids if eval(args.filter, {"p": pjobs[i-1]})]
        logging.info(f"Pjobs after filter: {len(pjob_ids)}")

    try:
        pjob_ids = pjob_ids[(args.jobid - 1) * args.reduction:args.jobid * args.reduction]
    except (AttributeError, TypeError):
        pass
    logging.info(f"My pjob ids: {int_to_str_ranges(pjob_ids)}")

    pjobs = [pjobs[i-1] for i in pjob_ids]

    return pjobs

def load_pjobs_from_json(file):
    with open(file) as f:
        return json.load(f, object_hook=pjob_json_decoder)

def load_pjobs(args, settings=None):
    if args.param_json is None:
        if args.param_file is None:
            if settings is None:
                raise Exception("load_pjobs: No param file or json given, unknown method")
            else:
                logging.info("load_pjobs: No param file or json given, trying to load from cmd args...")
                pjobs = [settings.from_cmdargs(args)]
        else:
            logging.warning("!!!!!!! DEPRECATED: Loading params from python")
            import importlib.util
            spec = importlib.util.spec_from_file_location("param", args.param_file)
            param_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(param_module)
            pjobs = param_module.pjobs
    else:
        pjobs = load_pjobs_from_json(args.param_json)

    pjobs = filter_pjobs(pjobs, args)

    for pjob in pjobs:
        pjob.update_from_cmdargs(args)

    return pjobs

def param_script_main(pjobs, unparsed_args=None):
    parser = argparse.ArgumentParser()
    add_pjobfilter_args(parser)
    parser.add_argument("--get")
    parser.add_argument("--len", action="store_true")
    parser.add_argument("--write-json", nargs="?", const="params.json")
    args = parser.parse_args(unparsed_args)
    pjobs = filter_pjobs(pjobs, args)

    if args.len:
        print(len(pjobs))
    elif args.write_json is not None:
        target = Path(args.write_json)
        if target.exists():
            raise Exception("not overwriting existing params.json")
        else:
            with open(target, "w") as file:
                pjobs_to_file(pjobs, file)
    else:
        for pjob in pjobs:
            if args.get is None:
                print(pjob)
            else:
                print(pjob[args.get])

add_param("method", default="PyRedLin", fixed=True)
add_param("tag", type=str)
add_param("pjob_id", default=1, type=int)
add_param("gtid", default=1, type=int)
add_param("gt_file", type=str)
add_param("num_mu_dirs", type=int, perf=True)
add_param("num_extra_nu_dirs", type=int, perf=True)
add_param("num_time_steps", default=3, type=int, perf=True)
add_param("fc", default=2, type=int, perf=True)
# add_param("res_root", default="res")
add_param("res_file", type=str)
add_param("recons_log", type=str, required=False)
add_param("eval_log", type=str, required=False)
add_param("projector", default="new", type=str, perf=True)
add_param("redcons_bound", default=0.001, type=float)
add_param("grid_size", default=100, type=int, perf=True)
add_param("noise_lvl", default=0.0, type=float)
add_param("data_scale", default=100.0, type=float, required=False)
# add_param("noise_const", default=0.2, type=float, required=False)
# add_param("data_scale", type=float, required=False)
add_param("noise_const", type=float, required=False)

add_adcg_param("method", default="ADCG", fixed=True)
add_adcg_param("tag", type=str)
add_adcg_param("pjob_id", default=1, type=int)
add_adcg_param("gtid", default=1, type=int)
add_adcg_param("gt_file", type=str)
add_adcg_param("num_time_steps", default=3, type=int, perf=True)
add_adcg_param("fc", default=2, type=int, perf=True)
# adadcg_d_param("res_root", default="res")
add_adcg_param("res_file", type=str)
add_adcg_param("recons_log", type=str, required=False)
add_adcg_param("eval_log", type=str, required=False)
add_adcg_param("noise_lvl", default=0.0, type=float)
add_adcg_param("max_outer_iter", default=100, type=int, perf=True)
add_adcg_param("max_cd_iter", default=200, type=int, fixed=True, perf=True)
add_adcg_param("space_grid_size", default=20, type=int, perf=True)
add_adcg_param("vel_grid_size", default=20, type=int, perf=True)
add_adcg_param("min_optim_gap", default=1e-5, type=float, fixed=True, perf=True)
add_adcg_param("min_obj_progress", default=1e-4, type=float, perf=True)
add_adcg_param("save_state", type=bool, default=False)
add_adcg_param("resume", type=bool, default=False)
