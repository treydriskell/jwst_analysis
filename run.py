import os.path as path
import subprocess


def run(command):
    print('Running ' + ' '.join(command))
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
    
    # run analysis, change n_jobs as you see fit
    command = ['python', 'analysis.py', 'paper_params', '--initial', '0', '--final'
               '73599', '--save', '--n_jobs', '1']
    run(command)

    command = ['python', 'plotting.py']
    run(command)

    command = ['python', 'astro_uvlf.py']
    run(command)

    command = ['python', 'plot_muvz_data.py']
    run(command)

    command = ['python', 'galform_comparison.py']
    run(command)

    command = ['python', 'plot_hst_uvlf.py']
    run(command)
    
    command = ['python', 'smf_comp.py']
    run(command)

    command = ['python', 'compute_chi2.py']
    run(command)

    command = ['python', 'test_jwst.py']
    run(command)

    

