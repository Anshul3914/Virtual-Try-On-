import argparse

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_type", type=str, default="normal", help="Weight initialization type")
    parser.add_argument("--init_variance", type=float, default=0.02, help="Weight initialization variance")
    return parser.parse_args()
