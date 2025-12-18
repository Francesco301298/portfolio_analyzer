import pandas as pd
import numpy as np
from scipy import stats

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
