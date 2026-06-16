# Copyright 2023 OmniSafeAI Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import atexit
import csv
import json
import os
import os.path as osp
import re
import warnings

import joblib
import numpy as np
from flax import nnx
from tensorboardX import SummaryWriter

color_2_num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38,
)


def _is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except Exception:
        return False


def _convert_json(obj):
    if _is_json_serializable(obj):
        return obj
    if isinstance(obj, dict):
        return {_convert_json(k): _convert_json(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [_convert_json(x) for x in obj]
    if hasattr(obj, "__name__") and "lambda" not in obj.__name__:
        return _convert_json(obj.__name__)
    if hasattr(obj, "__dict__") and obj.__dict__:
        return {
            str(obj): {
                _convert_json(k): _convert_json(v) for k, v in obj.__dict__.items()
            }
        }
    return str(obj)


def _colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color_2_num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    return "\x1b[{}m{}\x1b[0m".format(";".join(attr), string)


class Logger:
    def __init__(
        self,
        log_dir,
        seed=None,
        output_fname="progress.csv",
        level: int = 1,
        use_tensorboard: bool = True,
        verbose: bool = True,
    ):

        self.log_dir = log_dir
        self.level = level
        self.verbose = verbose
        os.makedirs(self.log_dir, exist_ok=True)

        self.output_file = open(
            os.path.join(self.log_dir, output_fname), encoding="utf-8", mode="w"
        )

        atexit.register(self.output_file.close)

        self._csv_writer = csv.writer(self.output_file)

        self.epoch = 0
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}

        parts = log_dir.rstrip("/").split("/")
        self.exp_name = "-".join(parts[-3:] + ["seed", str(seed)])
        self.use_tensorboard = use_tensorboard
        self.logged = True

        if use_tensorboard:
            self.summary_writer = SummaryWriter(os.path.join(self.log_dir, "tb"))

    def close(self):
        self.output_file.close()

    def log(self, msg, color="green"):
        if self.verbose and self.level > 0:
            print(_colorize(msg, color, bold=False))

    def log_tabular(self, key, val):
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, (
                f"New key '{key}' introduced after first iteration. "
                "Add it from the very first dump_tabular call."
            )
        assert key not in self.log_current_row, (
            f"Key '{key}' set twice before dump_tabular()."
        )
        self.log_current_row[key] = val

    def save_config(self, config):
        cfg = _convert_json(config)
        if self.exp_name is not None:
            cfg["exp_name"] = self.exp_name
        output = json.dumps(cfg, separators=(",", ":\t"), indent=4, sort_keys=True)
        with open(osp.join(self.log_dir, "config.json"), "w") as out:
            out.write(output)

    def nn_model_save(self, nn_model_saver_element, itr=None, prefix=""):
        """Serialise an NNX module to disk with joblib."""
        self.log("Saving model …")
        fpath = osp.join(self.log_dir, "nn_model_save")
        os.makedirs(fpath, exist_ok=True)
        fname = osp.join(
            fpath, prefix + "_model_" + ("%d" % itr if itr is not None else "") + ".pt"
        )
        _, state = nnx.split(nn_model_saver_element)
        joblib.dump(nnx.to_pure_dict(state), fname)
        self.log("Done.")

    def nn_model_load(self, model, model_path):
        """Restore an NNX module from a joblib file."""
        restored = joblib.load(model_path)
        graphdef, abstract_state = nnx.split(model)
        nnx.replace_by_pure_dict(abstract_state, restored)
        return nnx.merge(graphdef, abstract_state)

    def dump_tabular(self):
        """Print table, write CSV row, push to TensorBoard."""
        self.epoch += 1
        key_lens = [len(k) for k in self.log_headers]
        max_key_len = max(15, max(key_lens))
        keystr = "%" + "%d" % max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len

        if self.verbose and self.level > 0:
            print("-" * n_slashes)
        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            valstr = "%8.3g" % val if hasattr(val, "__float__") else val
            if self.verbose and self.level > 0:
                print(fmt % (key, valstr))
        if self.verbose and self.level > 0:
            print("-" * n_slashes, flush=True)

        if self.output_file is not None:
            if self.first_row:
                self._csv_writer.writerow(self.log_current_row.keys())
            self._csv_writer.writerow(self.log_current_row.values())
            self.output_file.flush()

        if self.use_tensorboard:
            for key, val in self.log_current_row.items():
                self.summary_writer.add_scalar(key, val, global_step=self.epoch)

        self.log_current_row.clear()
        self.first_row = False


class EpochLogger(Logger):
    """
    Extends Logger with epoch-level statistics.

    store()       – accumulate values during an epoch
    log_tabular() – log mean (and optionally min/max/std) at epoch end
    """

    def __init__(
        self,
        log_dir,
        seed=None,
        output_fname="progress.csv",
        level: int = 1,
        use_tensorboard: bool = True,
        verbose: bool = True,
    ):
        super().__init__(
            log_dir=log_dir,
            seed=seed,
            output_fname=output_fname,
            level=level,
            use_tensorboard=use_tensorboard,
            verbose=verbose,
        )
        self.epoch_dict = {}

    def dump_tabular(self):
        self.logged = True
        super().dump_tabular()

    def store(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.epoch_dict:
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

    def log_tabular(self, key, val=None, min_and_max=False, std=False):
        if val is not None:
            super().log_tabular(key, val)
        else:
            v = np.mean(self.epoch_dict[key])
            super().log_tabular(key, v)
            if min_and_max:
                super().log_tabular(key + "/Min", np.min(self.epoch_dict[key]))
                super().log_tabular(key + "/Max", np.max(self.epoch_dict[key]))
            if std:
                super().log_tabular(key + "/Std", np.std(self.epoch_dict[key]))
        self.epoch_dict[key] = []
