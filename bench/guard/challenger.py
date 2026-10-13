"""The challenger of the chooser: LambdaRank on the same rows, one query per site with KEEP included.

Its raw scores go through the same calibration by (op, kind) as the binary model's, so the two are compared at the same
operating point (`operating.report_by_op`); the labels inside a site are exclusive, which is what a ranking objective
expects, and the top-versus-KEEP margin is read from the calibrated rows like the binary model's.
"""

import lightgbm as lgb
import numpy as np

from extract.guard import features, model

CHALLENGER_CELL = (31, 200)


def lambdarank_learner(cell=CHALLENGER_CELL, seed: int = 1, threads: int = 2):
    """Factory for `train_chooser.run`: a LambdaRank booster fitted on the training sites (`groups` are their sizes)."""

    def fit(x: np.ndarray, y: np.ndarray, groups: np.ndarray):
        params = {
            **model.CHOOSER_FIXED,
            "objective": "lambdarank",
            "num_leaves": cell[0],
            "seed": seed,
            "num_threads": threads,
        }
        data = lgb.Dataset(
            x,
            label=y,
            group=groups,
            feature_name=list(features.FEATURES),
            categorical_feature=list(features.CATEGORICAL),
            free_raw_data=False,
        )
        booster = lgb.train(params, data, num_boost_round=cell[1])
        return lambda rows: booster.predict(rows, raw_score=True, num_threads=threads)

    fit.needs_groups = True
    return fit
