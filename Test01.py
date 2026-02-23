from edgar import company
from edgar import set_identity

set_identity("ncc", "alexbarrakett@gmail.com")

def main():
    c = company ("msft", "10-Q")