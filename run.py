"""
Master script to run the complete JWST analysis pipeline.

This script orchestrates the execution of all analysis and plotting scripts
in the correct order. It can optionally run the computationally expensive
analysis step, then generates all publication figures.
"""

import os.path as path
import subprocess
from argparse import ArgumentParser


def run(command: list, dryrun: bool = False) -> None:
    """Execute a shell command with error handling.

    Runs a command using subprocess and prints detailed output including
    stdout, stderr, and exit codes. Handles errors gracefully.

    Parameters
    ----------
    command : list
        List of strings representing the command and its arguments.
    dryrun : bool, optional
        If True, only prints the command without executing it.
        Default is False.

    Returns
    -------
    None
        Prints command output to console.
    """
    print('Running ' + ' '.join(command))
    if dryrun:
        return
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("Command executed successfully!")
        print("Standard Output:")
        print(result.stdout)
        print("Standard Error:")
        print(result.stderr)
        print(f"Exit Code: {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print("Standard Output (on error):")
        print(e.stdout)
        print("Standard Error (on error):")
        print(e.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Run complete JWST analysis pipeline")
    parser.add_argument("--dryrun", action='store_true', help="Do a dryrun")
    parser.add_argument("--run_analysis", action='store_true', help="Run the analysis pipeline")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs to run for analysis")
    args = parser.parse_args()

    # run analysis, change n_jobs as you see fit
    # analysis is optional since it can take a while
    if args.run_analysis:
        command = ['python', 'analysis.py', 'paper_params', '--base', 
                    '/carnegie/scidata/groups/dmtheory/jwst_simulated_data', 
                    '--initial', '0', '--final', '73599', '--save', '--n_jobs', f'{args.n_jobs}']
        run(command, args.dryrun)

    command = ['python', 'plotting/plotting.py']
    run(command, args.dryrun)

    command = ['python', 'plotting/astro_uvlf.py']
    run(command, args.dryrun)

    command = ['python', 'plotting/plot_muvz_data.py']
    run(command, args.dryrun)

    command = ['python', 'plotting/galform_comparison.py']
    run(command, args.dryrun)

    command = ['python', 'plotting/plot_hst_uvlf.py']
    run(command, args.dryrun)
    
    command = ['python', 'plotting/smf_comp.py']
    run(command, args.dryrun)

    command = ['python', 'compute_chi2.py']
    run(command, args.dryrun)

    command = ['python', 'test_jwst.py']
    run(command, args.dryrun)

    command = ['python', 'plotting/plot_appendix.py']
    run(command, args.dryrun)