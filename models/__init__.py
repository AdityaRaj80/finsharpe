# TimesNet was excluded from the finsharpe campaign because its FFT +
# Inception-block cost made every Track B run exceed the 12h SLURM walltime
# (verified across 4 jobs at H={5, 20, 60, 120} that all timed out without
# producing a Round-1 checkpoint). Documented in reports/design.md.
from .patchtst import Model as PatchTST
from .tft import Model as TFT
from .adapatch import Model as AdaPatch
from .gcformer import Model as GCformer
from .itransformer import Model as iTransformer
from .vanilla_transformer import Model as VanillaTransformer
from .dlinear import Model as DLinear

model_dict = {
    "PatchTST": PatchTST,
    "TFT": TFT,
    "AdaPatch": AdaPatch,
    "GCFormer": GCformer,
    "iTransformer": iTransformer,
    "VanillaTransformer": VanillaTransformer,
    "DLinear": DLinear,
}
