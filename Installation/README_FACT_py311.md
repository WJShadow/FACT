# FACT `fact_py311` environment installer

This installer installs only the Windows Conda/Python runtime environment. It does not copy FACT source code, models, experiment results, or other data files.

## Online package

Run `Install_FACT_Online_on_Win64.bat`. The package contains the locked YML, a local HDF5 package, the private-Miniconda bootstrap logic, and reference checksums. Miniconda, Conda packages, and the pinned PyPI wheels are downloaded from their configured repositories.

## Offline package

Download [FACT-fact_py311-offline.zip](https://drive.google.com/file/d/1cnJSZLHRWYwF34n71MbonNF1CHETFpzO/view?usp=drive_link), extract it with Windows **Extract All**, and run `Install_FACT_Offline_on_Win64.bat` from the extracted `FACT-fact_py311-offline` folder (or run the common `Install_FACT_on_Win64.bat`). The package carries the verified private Miniconda installer and a `conda-pack 0.9.2` archive produced from a validated `fact_py311` environment. Installation performs only checksum verification, extraction, prefix relocation, and runtime checks; it does not run Conda or pip downloads.

Both modes install to `%USERPROFILE%\.conda\envs\fact_py311`, register `Python (fact_py311)` as a Jupyter kernel, and write logs under `%LOCALAPPDATA%\FACT\logs`. Use `Activate_FACT.bat` to activate the environment without `conda init`.

The environment is intentionally pinned to match `fact_py311`; the source environment's single `imagecodecs`/NumPy `pip check` warning is treated as expected.
