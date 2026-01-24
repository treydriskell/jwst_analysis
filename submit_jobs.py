"""
Job submission script for running Galacticus simulations on HPC clusters.

This module generates SLURM job scripts and Galacticus XML input files for
parameter space exploration. It reads parameter configurations from YAML
files and creates batch jobs to run Galacticus simulations across the
parameter space.
"""

import numpy as np
import os
import os.path as path
import subprocess
import itertools
import xml.etree.ElementTree as ET
import yaml
from argparse import ArgumentParser


def create_xml(
    astro_params: dict, 
    outdir: str, 
    values: list, 
    param_paths: list, 
    z: str
) -> str:
    """Create a Galacticus XML input file for a parameter combination.

    Reads a template XML file and modifies parameter values, redshift, and
    output filenames according to the specified parameter combination.

    Parameters
    ----------
    astro_params : dict
        Dictionary containing configuration including 'xml_template' path.
    outdir : str
        Output directory for the XML and HDF5 files.
    values : list
        List of parameter values to set.
    param_paths : list
        List of XML paths (XPath-like) to parameter elements.
    z : str
        Redshift value as a string (e.g., '8.0').

    Returns
    -------
    str
        Path to the created XML file.
    """
    template_xml = astro_params['xml_template']
    tree = ET.parse(template_xml)
    root = tree.getroot()
    for j in range(len(values)):
        v = '{:.2e}'.format(values[j])
        param_path = param_paths[j]
        xml_ps = root.findall(param_path) 
        if len(xml_ps)==0:
            idx = param_path.rfind('/')
            parent = root.find(param_path[:idx])
            xml_p = ET.SubElement(parent, param_path[idx+1:], value=v)
        else:
            xml_ps[0].set('value', v) 
    basez = root.find('mergerTreeConstructor/redshiftBase')
    basez.set('value', z)
    outputFileName = root.find('outputFileName')
    out_fn = path.join(outdir, 'z'+z+'.hdf5')
    outputFileName.set('value', out_fn)
    outTimes = root.find('outputTimes/redshifts')
    outTimes.set('value', z)
    xml_fn = path.join(outdir, 'z'+z+'.xml')
    tree.write(xml_fn)
    return xml_fn


def create_jobs_from_list(
    args, 
    astro_params: dict, 
    xml_fns: list, 
    initial: int, 
    final: int
) -> str:
    """Create a SLURM job script from a list of XML files.

    Generates a SLURM batch job script that runs multiple Galacticus
    simulations. The job script is created from a template and includes
    commands to run Galacticus for each XML file.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing job_directory and template_job_file.
    astro_params : dict
        Dictionary containing 'param_dir' for naming.
    xml_fns : list
        List of XML file paths to include in the job.
    initial : int
        Starting parameter index (for job naming).
    final : int
        Ending parameter index (for job naming).

    Returns
    -------
    str
        Path to the created job script file.
    """
    template_job = args.template_job_file
    fn_base = astro_params['param_dir']+'_pi{}_pf{}'.format(initial,final)
    with open(template_job, 'r') as f:
        lines = f.readlines()
    lines[6] = '#SBATCH --job-name=' + fn_base + '\n'
    lines[7] = f'#SBATCH --output={args.job_directory}' + fn_base + '.out\n'
    lines[8] = f'#SBATCH --error={args.job_directory}' + fn_base + '.err\n'
    for xml_fn in xml_fns:
        right = xml_fn.rfind('.') 
        out_fn = xml_fn[:right]+'.out'
        lines.append('/home/gdriskell/Galacticus/galacticus/Galacticus.exe '+ xml_fn + ' &> ' + out_fn  +'\n')
    lines.append('echo "job ended at `date`" \n')
    lines.append('exit\n')
    job_fn = path.join(args.job_directory, fn_base + '.job')
    with open(job_fn, 'w') as f:
        f.writelines(lines)
    return job_fn


def run(args) -> None:
    """Main function to generate and submit Galacticus simulation jobs.

    Reads parameter configuration from YAML, generates XML input files for
    all parameter combinations, creates SLURM job scripts, and optionally
    submits them to the queue.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments including:
        - yaml_file: Path to YAML parameter configuration
        - job_directory: Directory for job scripts
        - output_directory: Directory for simulation outputs
        - template_job_file: Path to SLURM job template
        - dryrun: If True, don't submit jobs
        - n_params_per_job: Number of parameter combinations per job

    Returns
    -------
    None
        Creates XML files and job scripts, optionally submits jobs.
    """
    yaml_fn = args.yaml_file

    with open(yaml_fn, 'r') as f:
        astro_params = yaml.safe_load(f)

    pvalues = []
    parameters = []
    param_paths = []
    for k,p in astro_params['parameters'].items():
        if not p['sample']:
            values = [p['value']]
        elif p['sample']=='lin':
            values = np.linspace(p['min'], p['max'], p['nv'])
        elif p['sample']=='log':
            values = np.geomspace(float(p['min']), float(p['max']), p['nv'])
        else:
            raise Exception('Unknown value for sample')
        pvalues.append(values)
        parameters.append(k)
        param_paths.append(p['path'])
    
    params_per_job = args.n_params_per_job

    metadata = []

    xml_fns = []
    jobids = []
    for i, values in enumerate(itertools.product(*pvalues)):
        if i == 0:
            j = 0

        fn_base = astro_params['param_dir']+'_p{}'.format(j)
        outdir = path.join(args.output_directory, fn_base)

        if not path.isdir(outdir):
            os.makedirs(outdir)

        for z in astro_params['zs']:
            xml_fn =  create_xml(astro_params, outdir, values, param_paths, z) #path.join(outdir, 'z'+z+'.xml') 
            xml_fns.append(xml_fn)

        if (i%params_per_job) == (params_per_job-1):
            os.chdir(args.job_directory)
            job_fn = create_jobs_from_list(args, astro_params, xml_fns,j,i)
            print(f'Creating job file : {job_fn}')
            if not args.dryrun:
                output = subprocess.check_output(['sbatch', job_fn], text=True)
    
            xml_fns = []
            j = i + 1
                
    if len(xml_fns)>1:
        os.chdir(args.job_directory)
        if not args.dryrun:
            job_fn = create_jobs_from_list(args, astro_params, xml_fns,i-(i%params_per_job),i)
            os.system('sbatch ' + job_fn)
    

if __name__ == "__main__":
    parser = ArgumentParser(description="")
    
    parser.add_argument("--yaml_file", type=str, help="Path to yaml file for parameters to run. " \
    "Example : /home/gdriskell/jwst_analysis/yamls/test_params.yaml ")
    parser.add_argument("--job_directory", type=str,  help="Path to directory to save job files. " \
    "Example: /carnegie/nobackup/users/gdriskell/jwst_data/jobs/") 
    parser.add_argument("--output_directory", type=str, help="Path to directory where output should be saved." \
    "Example: /carnegie/scidata/groups/dmtheory/jwst_simulated_data")
    parser.add_argument("--template_job_file", type=str,  help="Path to directory where output should be saved." \
    "Example: /carnegie/scidata/groups/dmtheory/jwst_simulated_data")
    parser.add_argument("--dryrun", action='store_true', help="Do a dryrun")
    parser.add_argument("--n_params_per_job", type=int, default=1, help="Number of parameters per job run")
    args = parser.parse_args()
    run(args)