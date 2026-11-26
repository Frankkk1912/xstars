# XSTARS

Excel and WPS statistics and publication figures for wet-lab assays: Western blot (WB), qPCR, CCK-8, and ELISA. XSTARS picks a statistical test, draws significance brackets, and saves a journal-style figure.

This is not the astronomy code [XSTAR](https://heasarc.gsfc.nasa.gov/xstar/xstar.html).

Most people should not install from PyPI. Download the Windows, macOS, or WPS installer from [GitHub Releases](https://github.com/Frankkk1912/xstars/releases). Those packages bundle Python and the spreadsheet add-in. No separate Python environment is required.

## Install from PyPI

Python users who want the statistics and plotting engine:

```bash
pip install xstars
```

That install is pandas, numpy, scipy, scikit-posthocs, matplotlib, seaborn, and Pillow. It does not install Excel or the settings dialogs.

The Excel/WPS UI (xlwings and ttkbootstrap) is an optional extra:

```bash
pip install "xstars[excel]"
```

`xstars[excel]` is for Windows and macOS spreadsheet hosts. The core package imports without xlwings, ttkbootstrap, or tkinter.

## Example

```python
import pandas as pd

from xstars.application.analysis import analyze_dataframe
from xstars.config import PrismConfig
from xstars.plot_engine import PlotEngine

df = pd.DataFrame(
    {
        "Control": [1.1, 1.2, 1.0, 1.3, 1.15, 1.25],
        "Treatment": [2.1, 2.4, 1.9, 2.2, 2.0, 2.3],
    }
)
config = PrismConfig(title="Treatment vs control", y_label="Signal")
result = analyze_dataframe(df, config)
figure = PlotEngine(config).plot(result.transformed_data, result.stats_result)
figure.savefig("xstars_figure.png", dpi=200)
```

`analyze_dataframe` runs the shared transform, statistics, and plot pipeline. `PlotEngine` draws the figure that this example writes to `xstars_figure.png`.

## Links

- Source and full documentation: https://github.com/Frankkk1912/xstars
- 中文文档: https://github.com/Frankkk1912/xstars/blob/main/README.zh-CN.md
- Installers: https://github.com/Frankkk1912/xstars/releases
- Issues: https://github.com/Frankkk1912/xstars/issues
