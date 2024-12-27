import sys

import requests

if __name__ == "__main__":
    if sys.version_info >= (3, 11):
        print("Hello World!")
    else:
        print("requires Python version >= 3.11")
