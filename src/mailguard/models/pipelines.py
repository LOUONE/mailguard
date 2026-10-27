"""The two Phase 1 baselines, each a single scikit-learn Pipeline from `list[Email]` to a score.

Keeping parsing out and everything else in means the exact object trained here is the one the API
loads in Phase 2: there is no separate feature code to keep in sync.
"""

from __future__ import annotations

from typing import Final, Literal, get_args

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from mailguard.features.headers import AttachmentFeatures, HeaderFeatures
from mailguard.features.text import LexiconFeatures, TextFeatures
from mailguard.features.urls import UrlFeatures
from mailguard.models.config import BaselineConfig, TextConfig

ModelName = Literal["logreg", "lightgbm"]
MODEL_NAMES: Final[tuple[ModelName, ...]] = get_args(ModelName)

# Types that skops may rebuild when a saved pipeline is loaded, beyond the scikit-learn and numpy
# types it trusts already. Loading never runs code from the file (unlike pickle), so a tampered
# model in the registry cannot execute anything. A test checks this list covers both pipelines.
TRUSTED_TYPES: Final = (
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier",
    "mailguard.features.headers.AttachmentFeatures",
    "mailguard.features.headers.HeaderFeatures",
    "mailguard.features.text.LexiconFeatures",
    "mailguard.features.text.TextFeatures",
    "mailguard.features.urls.UrlFeatures",
)


def _text(config: TextConfig) -> TextFeatures:
    return TextFeatures(
        word_max_features=config.word_max_features,
        char_max_features=config.char_max_features,
        min_df=config.min_df,
    )


def logreg_pipeline(config: BaselineConfig) -> Pipeline:
    """Baseline 1: logistic regression on body-text TF-IDF only."""
    return Pipeline(
        [
            ("features", FeatureUnion([("text", _text(config.logreg.text))])),
            (
                "classifier",
                LogisticRegression(
                    C=config.logreg.C, max_iter=config.logreg.max_iter, random_state=config.seed
                ),
            ),
        ]
    )


def lightgbm_pipeline(config: BaselineConfig) -> Pipeline:
    """Baseline 2: gradient-boosted trees on every feature group."""
    params = config.lightgbm
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("headers", HeaderFeatures()),
                        ("attachments", AttachmentFeatures()),
                        ("urls", UrlFeatures()),
                        ("lexicon", LexiconFeatures()),
                        ("text", _text(params.text)),
                    ]
                ),
            ),
            (
                "classifier",
                LGBMClassifier(
                    n_estimators=params.n_estimators,
                    learning_rate=params.learning_rate,
                    num_leaves=params.num_leaves,
                    min_child_samples=params.min_child_samples,
                    colsample_bytree=params.colsample_bytree,
                    reg_lambda=params.reg_lambda,
                    n_jobs=params.n_jobs,
                    random_state=config.seed,
                    deterministic=True,
                    force_row_wise=True,
                    verbose=-1,
                ),
            ),
        ]
    )


def build_pipeline(name: ModelName, config: BaselineConfig) -> Pipeline:
    return logreg_pipeline(config) if name == "logreg" else lightgbm_pipeline(config)
