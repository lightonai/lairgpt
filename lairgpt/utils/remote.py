import os
import wget
import json
import errno
import shutil
import socket
import tarfile
import logging
import urllib.request

DISCLAIMER = (
    "\nDISCLAIMER: PAGnol is made available under the MIT licence: "
    "By downloading its models/assets, you agree with the terms of the licence agreement.\n"
    "Under no circumstances will LightOn and/or Inria be held responsible or liable"
    " in any way for any claims, damages, losses, expenses, costs or liabilities "
    "whatsoever (including, without limitation, any direct or indirect damages for "
    "loss of profits, business interruption or loss of information) resulting or "
    "arising directly or indirectly from your use of or inability to use PAGnol.\n"
    "Use judgement and discretion before deploying PAGnol: you are responsible for "
    "using it thoughtfully and responsibly, in a way that benefits society.\n"
)

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
local_dir = os.path.expanduser("~/.cache/lairgpt/")
remote_dir = "https://download.lighton.ai/pagnol_ccnet/latest/"
version_file = os.path.join(local_dir, "versions.json")

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
    connected = is_connected()
    logged = is_logged(filename, alt)

    if connected:
        local_hash, remote_hash = "", ""
        if logged:
            with open(version_file) as local_vers:
                local_hash = json.load(local_vers)[filename]
            with urllib.request.urlopen(remote_dir + "versions.json") as remote_vers:
                remote_hash = json.loads(remote_vers.read().decode())[filename]
        if local_hash == remote_hash != "" and os.path.isfile(dest):
            logging.info("Latest version of the " + alt + " available.")
        else:
            logging.warn(
                "It seems you don't have the latest version " + \
                "of the " + alt + " on your local machine."
            )

            # Printing the License Disclaimer
            # if downloading asset for te first time
            if resp is None and not os.path.isfile(dest):
                print(DISCLAIMER)

            if resp or resp is None and query_yes_no(
                    "Would you like to download the " + alt + " from LAIR repos?"
            ):
                try:
                    os.remove(dest)
                except:
                    pass
                dest = download_latest(filename)
            else:
                logging.warn("Falling back to a local version if available.")
    else:
        logging.warn(
            "CAN'T CONNECT TO REMOTE HOST -- Cannot verify the lib's assets versions."
        )

    if not os.path.isfile(dest):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), dest)

    return dest


def is_logged(filename, alt):
    """Utility to check versioning state of
    the lib's cache for an asset file

    Parameters
    ----------
    filename: str
        Required asset filename.
    alt: str
        Required asset descriptor.

    Returns
    -------
    bool
        True if versions.json exists already, False otherwise.
    """
    if not os.path.isfile(version_file):
        logging.warn(
            "VERSIONING METADATA NOT FOUND -- Cannot verify the" + \
            "lib's assets versions.\n" + \
            "Creating it at " + version_file
        )
        with open(version_file, 'w') as outdump:
            json.dump({}, outdump)
        return False
    else:
        with open(version_file) as versions:
            if not filename in json.load(versions):
                logging.warn(
                    "The " + alt + "'s version is not stated in versioning metadata."
                )
                return False
    return True


def download_latest(filename):
    """Utility to download latest version of an asset file
    from LAIR's remote repo and update cache's versioning

    Parameters
    ----------
    filename: str
        Name of the file to download.

    Returns
    -------
    str
        path to the new downloaded file
    """
    # download asset file
    url = remote_dir + filename + ".tar.gz"
    dest = os.path.join(local_dir, filename)
    os.makedirs(local_dir, exist_ok=True)
    wget.download(url, dest + ".tar.gz")

    # extract |> remove archive
    with tarfile.open(dest + ".tar.gz", "r:gz") as tar:
        tar.extractall(local_dir)
    os.remove(dest + ".tar.gz")
    print("\nDownloaded at " + dest)

    # Update asset's hash in lib's cache versions
    versions = {}
    with open(version_file) as dump:
        versions = json.load(dump)
    os.remove(version_file)
    with urllib.request.urlopen(remote_dir + "versions.json") as remote_vers:
        remote_hash = json.loads(remote_vers.read().decode())[filename]
    versions[filename] = remote_hash
    with open(version_file, 'w') as outdump:
        json.dump(versions, outdump)
    logging.info("Cache versioning log updated at " + local_dir + "versions.json")
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
