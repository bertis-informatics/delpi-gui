import sys
import os
import argparse

os.environ["PYTHONUTF8"] = "1"
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

current_dir = os.path.dirname(os.path.abspath(__file__))
delpi_path = os.path.join(current_dir, "delpi")

pymsio_path = os.path.join(delpi_path, "pymsio")

sys.path.insert(0, pymsio_path)
sys.path.insert(0, delpi_path)

from delpi.run import run_search

if __name__ == "__main__":
    # Parse the config file path passed from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config yaml")
    args = parser.parse_args()

    print(f"Engine Starting with Config: {args.config}")
    
    # Execute the DelPi engine
    run_search(config_path=args.config, device="cuda:0", log_level="info")