# JWST Analysis Repository

This repository contains analysis scripts and plotting tools for processing JWST (James Webb Space Telescope) simulated data from Galacticus and generating figures for scientific publication.

## Overview

The repository processes simulated galaxy data from Galacticus simulations, performs statistical analysis, and generates publication-quality figures. The analysis includes:

- Processing of Galacticus HDF5 output files
- Statistical analysis of galaxy properties (magnitudes, UV luminosity functions)
- Parameter space exploration and best-fit identification
- Generation of comparison plots with observational data
- Statistical tests (e.g., Peacock test via ndtest)

## Installation

### Prerequisites

- Python 3.10
- Conda (recommended) or pip

### Setup

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd jwst_analysis
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate jwst
   ```

   Alternatively, install dependencies manually:
   ```bash
   pip install h5py matplotlib numpy pandas scipy seaborn astropy cmasher
   ```

3. (Optional) For statistical tests, install `ndtest`:
   ```bash
   pip install ndtest
   # OR
   git clone https://github.com/syrte/ndtest
   cd ndtest
   pip install .
   ```

## Data Structure

The raw data is hosted on OBS in the directory `.../jwst_simulated_data`. The data structure is organized as follows:

- Each parameter combination is saved in a separate subdirectory: `paper_params_p{i}` where `i` runs from 0 to 73599
- The best-fit parameter from the analysis is 13845, located in `paper_params_p13845`
- For each parameter set, Galacticus outputs:
  - Input parameter file: `paper_params_p{i}/z{z}.xml`
  - Data file: `paper_params_p{i}/z{z}.hdf5`
  - Where `z ∈ ["8.0", "12.0", "16.0"]` (e.g., `paper_params_p0/z8.0.xml`)

The analysis script processes data from the HDF5 files and saves data products to the same subdirectory.

## Usage

### Quick Start

To run the complete analysis pipeline and generate all figures:

```bash
python run.py --run_analysis --n_jobs <number_of_cores>
```

This will:
1. Run the analysis on all parameter combinations (optional, can be time-consuming)
2. Generate all publication figures
3. Compute chi-squared statistics
4. Run statistical tests

### Individual Scripts

#### Analysis

**`analysis.py`** - Main analysis script that processes Galacticus output files.

Processes all parameter combinations and saves results to CSV:
```bash
python analysis.py paper_params --initial 0 --final 73599 --save --n_jobs <n>
```

Use `--help` to see all available arguments:
```bash
python analysis.py --help
```

**Note:** Parallelization uses joblib. Running over multiple nodes has not been tested, so it's recommended to use 1 node and cap the number of jobs at the number of CPU cores.

#### Plotting Scripts

Each script generates specific figures for the publication:

- **`plotting.py`** - Generates Figures 2, 4, 6, and 8
- **`astro_uvlf.py`** - Generates Figure 3 (UV luminosity function)
- **`plot_muvz_data.py`** - Generates Figure 5 (magnitude-redshift data)
- **`galform_comparison.py`** - Generates Figure 7 (Galform comparison)
- **`plot_hst_uvlf.py`** - Generates left panel of Figure 9 (HST UVLF)
- **`smf_comp.py`** - Generates right panel of Figure 9 (stellar mass function comparison)
- **`plot_appendix.py`** - Generates appendix figures

**Important:** Scripts that require the best-fit parameter have the index hard-coded (currently 13845). If the best-fit index changes, you'll need to update these indices manually in the relevant scripts.

#### Statistical Analysis

- **`compute_chi2.py`** - Computes chi-squared statistics for model comparison
- **`test_jwst.py`** - Runs Peacock test (requires `ndtest` package)

#### Job Submission (HPC)

**`submit_jobs.py`** - Generates and submits SLURM job scripts for running Galacticus simulations on HPC clusters.

Example usage:
```bash
python submit_jobs.py \
    --yaml_file yamls/paper_params.yaml \
    --job_directory /path/to/jobs/ \
    --output_directory /path/to/output/ \
    --template_job_file jwst_template.job \
    --n_params_per_job 1
```

## Project Structure

```
jwst_analysis/
├── analysis.py              # Main analysis script
├── run.py                   # Master script to run entire pipeline
├── submit_jobs.py           # HPC job submission script
├── compute_chi2.py          # Chi-squared computation
├── test_jwst.py             # Statistical tests
│
├── plotting.py              # Figures 2, 4, 6, 8
├── astro_uvlf.py            # Figure 3
├── plot_muvz_data.py        # Figure 5
├── galform_comparison.py    # Figure 7
├── plot_hst_uvlf.py         # Figure 9 (left)
├── smf_comp.py              # Figure 9 (right)
├── plot_appendix.py         # Appendix figures
│
├── data/                    # Observational data and processed files
│   ├── CEERS_data.csv       # CEERS survey data
│   ├── ngdeep_data.csv      # NGDEEP survey data
│   ├── obs.txt              # UniverseMachine data (see attribution)
│   ├── zgrid_hmfs.hdf5      # Halo mass function grid
│   └── zgrid_weights.npy    # Redshift grid weights
│
├── yamls/                   # Parameter configuration files
│   ├── paper_params.yaml    # Main parameter set
│   ├── smf_params.yaml      # Stellar mass function parameters
│   └── test_lower_tau0.yaml # Test parameter set
│
├── xmls/                    # Galacticus XML templates
│   ├── timescale_template.xml
│   └── zgrid_hmf.xml
│
├── jwst_template.job        # SLURM job template
├── environment.yml          # Conda environment specification
└── README.md               # This file
```

## Configuration

Parameter sets are defined in YAML files in the `yamls/` directory. These files specify:
- Parameter ranges and sampling methods (linear/log)
- Redshift values to simulate
- XML template paths
- Output directory names

## Data Attribution

- **`data/obs.txt`**: This file is from [UniverseMachine](https://bitbucket.org/pbehroozi/universemachine/src/main/) and is included for self-contained analysis. Any use of this data should credit the original authors and paper.

## Troubleshooting

1. **Best-fit index changed**: If the best-fit parameter index changes from 13845, search for hard-coded references to this index in plotting scripts and update them.

2. **Missing dependencies**: Ensure all packages from `environment.yml` are installed. Some scripts may require additional packages not listed (e.g., `ndtest` for statistical tests).

3. **Path issues**: Some scripts contain hard-coded paths (e.g., `/carnegie/scidata/...`). Update these to match your system's directory structure.

4. **Memory issues**: For large parameter spaces, consider processing in batches by adjusting the `--initial` and `--final` arguments in `analysis.py`.

## Citation

If you use this code or data in your research, please cite the relevant papers and acknowledge the original data sources (UniverseMachine, CEERS, NGDEEP surveys).
