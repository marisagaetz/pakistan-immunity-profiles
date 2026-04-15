# pakistan-immunity-profiles

This repository is intended to serve as a partial update or complement to https://github.com/NThakkar-IDM/intensification/. Each file works towards the eventual generation of province-level and national measles immunity profiles for Pakistan (SurvivalPrior.py).

Methodological updates are summarized in Immunity_Profile_Methods_Updates.pdf. Updates to the intensification codebase include these methodological changes, as well as the following:
1. Unified processing of DHS and MICS data (demography/survey_io.py).
2. Rewrites of demography code to utilize statsmodels machinery in favor of custom regression code.
3. Automatic extrapolation (and optional rescaling) of estimated birth rate and MCV coverage trends based on national World Bank and WHO estimates (extrapolate_trends.py).
4. Compilation of demography outputs into datasets that can be fed seamlessly into SurvivalPrior.py (CompileDatasets.py).
5. Accounting of province- and age- eligibility for SIAS (CompileDatasets.py) that is incorporated into a timeline of vaccination opportunities based on semimonth of birth, allowing SIAS to sit pre-MCV1 age as well as between MCV1 and MCV2 ages (survival_prior_core.py).

This repository uses the same environment.yml as the intensification codebase, and assumes the user already has the following:
1. A linelist of cases with age in months, date of onset, number of vaccine doses received, province, final status (lab-confirmed, epi-linked, discarded, clinically compatible). This linelist should be regressed using the intensification/methods/epi_curves workflow, so that clinically compatible cases get assigned a confirmation probability. This regressed linelist should be put in the pickle_jar directory.
2. DHS and/or MICS survey data in the _data/_survey directory. Note that survey_io.py requires specific file names and structures (see demography/survey_io.py for more detail).
3. World Bank national population and birth rate estimates in the _data repository.
4. WHO national MCV1 and MCV2 coverage estimates in the _data repository. 
