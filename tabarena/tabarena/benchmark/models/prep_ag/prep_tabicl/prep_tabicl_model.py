from __future__ import annotations

from tabarena.benchmark.models.ag.tabicl.tabicl_model import TabICLv2Model
from tabarena.benchmark.models.prep_ag.prep_mixin import ModelAgnosticPrepMixin


class PrepTabICLv2Model(ModelAgnosticPrepMixin, TabICLv2Model):
    ag_key = "prep_TABICL-V2"
    ag_name = "prep_TabICL-v2"
