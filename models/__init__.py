# TimesNet was excluded from the finsharpe campaign because its FFT +
# Inception-block cost made every Track B run exceed the 12h SLURM walltime
# (verified across 4 jobs at H={5, 20, 60, 120} that all timed out without
# producing a Round-1 checkpoint). Documented in reports/design.md.
#
# Under PLAN_v2, we focus on 4 main "modern" backbones (PatchTST, TFT,
# GCFormer, DLinear) PLUS 3 simple classical baselines (LSTM, RNN, CNN)
# to land the "simple-vs-complex" axis of the benchmark contribution.
# AdaPatch, iTransformer, VanillaTransformer remain available for
# supplementary tables but are NOT in the headline 7-model panel.
from .patchtst import Model as PatchTST
from .tft import Model as TFT
from .adapatch import Model as AdaPatch
from .gcformer import Model as GCformer
from .itransformer import Model as iTransformer
from .vanilla_transformer import Model as VanillaTransformer
from .dlinear import Model as DLinear
from .lstm import Model as LSTM
from .rnn import Model as RNN
from .cnn import Model as CNN

model_dict = {
    # --- Main 4 ("modern, complex" arm of the benchmark) ---
    "PatchTST": PatchTST,
    "TFT": TFT,
    "GCFormer": GCformer,
    "DLinear": DLinear,
    # --- Simple 3 ("classical" arm of the benchmark) ---
    "LSTM": LSTM,
    "RNN": RNN,
    "CNN": CNN,
    # --- Supplementary (not in headline panel) ---
    "AdaPatch": AdaPatch,
    "iTransformer": iTransformer,
    "VanillaTransformer": VanillaTransformer,
}

# Headline 7-model panel (kept stable across the paper):
HEADLINE_MODELS = ["PatchTST", "TFT", "GCFormer", "DLinear", "LSTM", "RNN", "CNN"]
