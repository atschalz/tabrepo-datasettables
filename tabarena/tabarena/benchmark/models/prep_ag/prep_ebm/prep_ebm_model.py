from __future__ import annotations

from tabarena.benchmark.models.ag.ebm.ebm_model import ExplainableBoostingMachineModel
from tabarena.benchmark.models.prep_ag.prep_mixin import ModelAgnosticPrepMixin


class PrepEBMModel(ModelAgnosticPrepMixin, ExplainableBoostingMachineModel):
    ag_key = "prep_EBM"
    ag_name = "prep_ExplainableBM"
