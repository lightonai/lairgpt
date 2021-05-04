import os
import wget
import json
import errno
import shutil
import socket
import tarfile
import urllib.request

local_dir = os.path.expanduser("~/.cache/lairgpt/")
remote_dir = "https://download.lighton.ai/pagnol_ccnet/latest/"


def load_asset(path, alt, resp=None):
    """Utility to assert model-related assets existence

    Parameters
    ----------
    path: str
        Asset file path to load.
    alt: str
        Asset file descriptor.
    resp: bool | None
        Default response for download/update prompts.
        if None, fall back to CLI-prompt.

    Returns
    -------
    str
        Asset's correct path (after download if non-existent)
    """
    dest = os.path.expanduser(path)
    filename = os.path.basename(dest)
    if not os.path.isfile(dest):
        print("It seems you don't have the " + alt + " on your local machine.")
        if resp is None and query_yes_no(
            "Would you like to download the " + alt + " from LAIR repos?"
        ):
            dest = download_latest(filename)
        elif resp:
            print("Downloading the " + alt + " from LAIR repos confirmed!")
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), dest)
    elif not is_connected():
        print("CAN'T CONNECT TO LAIR'S REPOS -- Cannot verify the " + alt + " version.")
    elif not os.path.isfile(os.path.join(local_dir, "versions.json")):
        print("CAN'T FIND LOCAL VERSIONING METADATA -- Cannot verify the " + alt + " version.")
    else:
        local_hash, remote_hash = "", ""
        with open(os.path.join(local_dir, "versions.json")) as local_vers:
            local_hash = json.load(local_vers)[filename]
        with urllib.request.urlopen(remote_dir + "versions.json") as remote_vers:
            remote_hash = json.loads(remote_vers.read().decode())[filename]
        if local_hash == remote_hash:
            print("Latest version of the " + alt + " available.")
        else:
            if query_yes_no("Would you like to download the latest version of the " + alt + ""):
                os.remove(dest)
                dest = download_latest(filename)
            else:
                print("Falling back to outdated local version.")
    return dest


def download_latest(filename):
    """Utility to download latest version of
    an asset file from LAIR's remote repo

    Parameters
    ----------
    filename: str
        Name of the file to download.

    Returns
    -------
    str
        path to the new downloaded file
    """
    url = remote_dir + filename + ".tar.gz"
    dest = os.path.join(local_dir, filename)
    os.makedirs(local_dir, exist_ok=True)
    wget.download(url, dest + ".tar.gz")
    with tarfile.open(dest + ".tar.gz", "r:gz") as tar:
        tar.extractall(local_dir)
    os.remove(dest + ".tar.gz")
    print(" Downloaded")
    return dest


def clear_cache(folder=local_dir):
    """Utility to clear cache folder

    Parameters
    ----------
    folder: str
        Path to cache directory. Defaults to already defined local_dir variable.
    """
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def is_connected(hostname="1.1.1.1"):
    """Validates internet connection.

    Parameters
    ----------
    hostname: str
        hostname to test reachability against.

    Returns
    -------
    bool
        True if reachable (Internet connection is ON), False otherwise.
    """
    try:
        # assert existence of a listening DNS
        host = socket.gethostbyname(hostname)
        # assert host is reachable
        s = socket.create_connection((host, 80), 2)
        s.close()
        return True
    except:
        pass
    return False


def query_yes_no(question, default=True):
    """Ask a yes/no question via input().

    Parameters
    ----------
    question: str
        prompt presented to the user.
    default: bool
        presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    Returns
    -------
    bool
        True if agreed, False otherwise.
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    assert default in (True, False, None)
    if default is None:
        prompt = " [y/n] "
    elif default:
        prompt = " [Y/n] "
    elif not default:
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        choice = input(question + prompt).lower()
        if default is not None and choice == "":
            return default
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' " "(or 'y' or 'n').")
