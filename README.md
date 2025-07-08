# Working with the raw data:
The data is currently hosted on OBS at the directory `/carnegie/scidata/groups/dmtheory/jwst_simulated_data`. The raw data for each parameter combination is saved to a seperate subdirectory labeled `.../jwst_simulated_data/paper_params_p{i}` where i runs from 0 to 73599. The best-fit parameter from our analysis for example is 13845 so is located in subdirectory `paper_params_p13845`. The relevant outputs from galacticus are saved as `paper_params_p{i}/z{z}.xml` (for the input parameter file) and `paper_params_p{i}/z{z}.hdf5` (for the data), where z $\in$ ["8.0", "12.0", "16.0"], e.g. `paper_params_p0/z8.0.xml`. The analysis script will process the data from the hdf5 file and save data products to the same subdirectory.

To run everything needed for the paper simply execute `run.py`. Below are details for what each script does to aid in the process of debugging if anything goes wrong.

## Script Details 

analysis.py loops through files and performs all the necessary analysis, saving results to <outfilename>.csv. Use "python analysis.py --help" to see which arguments need to be passed running analysis. If running for the first time using the raw data, it is recommended to run:

python analysis.py paper_params --initial 0 --final 73599 --save --n_jobs X

Parallelization is done using joblib, running over multiple nodes has not been tested, so it recommended to use 1 node and cap number of jobs at # of cores.

Plots for the figure are generated as follows:
- Figure 2,4,6,8: run plotting.py
- Figure 3: run astro_uvlf.py 
- Figure 5: run plot_muvz_data.py
- Figure 7: run galform_comparison.py
- Figure 9: run plot_hst_uvlf.py (left) and smf_comp.py (right)

Note that any plot that requires a best fit has the index hard-coded for now. If the best fit index were to change from 13845, you need to go in and change those indices by hand when loading the data.

To run the Peacock test results, you need to download [ndtest](https://github.com/syrte/ndtest) and either run `test_jwst.py` from within ndtest or pip install it as a package.
