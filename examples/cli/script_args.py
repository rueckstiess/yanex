# script.py
import sys
import yanex

# Get yanex parameters
params = yanex.get_params()

# Get script arguments
script_args = sys.argv[1:]  # Arguments after --
verbose = "--verbose" in script_args

print(f"Script arguments: {script_args}")
print(f"Verbose mode: {verbose}")
