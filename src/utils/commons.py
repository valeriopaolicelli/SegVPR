import torch
import numpy as np
import random
import os
import sys
import logging
import traceback
import psutil
import time
import socket


def set_paths_ws(args):
    hostname = socket.gethostname()

    src_folder = os.path.abspath(os.curdir).split('iccv2021/')[1].split('/')[0]

    # roots for hermes, phoenix and vandal cluster
    root, root_dataset_root = '/home/valerio/', '/home/valerio/datasets/'

    if 'kronos' in hostname:
        root, root_dataset_root = '/data/users/valerio/', '/data/datasets/valerio/'
    elif ('gnode' in hostname) or ('fnode' in hostname) or ('all' in hostname):
        my_dir = 'cmasone/valerio' if 'cmasone' in os.path.abspath(os.curdir) else 'vpaolicelli'
        root, root_dataset_root = f'/work/{my_dir}/', f'/work/{my_dir}/datasets/'

    args.output_folder = os.path.join(root, 'iccv2021', src_folder, args.output_folder)
    args.dataset_root = os.path.join(root_dataset_root, args.dataset_root.split('/home/valerio/datasets/')[1])
    args.dataset_root_val = os.path.join(root_dataset_root, args.dataset_root_val.split('/home/valerio/datasets/')[1])
    args.dataset_root_test = os.path.join(root_dataset_root, args.dataset_root_test.split('/home/valerio/datasets/')[1])
    args.DA_datasets = os.path.join(root_dataset_root, args.DA_datasets.split('/home/valerio/datasets/')[1])

    args.path_model = os.path.join(root, 'models')

    return args


def pause_while_running(pid):
    """Se il PID è -1 parte subito"""
    print(f'Sono il processo {os.getpid()}', end=' ')
    if int(pid) != -1:
        print(f"e sto aspettando il processo {pid} ...")
    while int(pid) in psutil.pids():
        time.sleep(5)
    print(f"e ora parto")


def make_deterministic(seed=0):
    # Make results deterministic. If seed == -1, do not make deterministic.
    # Running the script in a deterministic way might slow it down.
    if seed == -1:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(output_folder, console="debug",
                  info_filename="info.log", debug_filename="debug.log"):
    """Set up logging files and console output.
    Creates one file for INFO logs and one for DEBUG logs.
    Args:
        output_folder (str): creates the folder where to save the files.
        debug (str):
            if == "debug" prints on console debug messages and higher
            if == "info"  prints on console info messages and higher
            if == None does not use console (useful when a logger has already been set)
        info_filename (str): the name of the info file. if None, don't create info file
        debug_filename (str): the name of the debug file. if None, don't create debug file
    """
    if os.path.exists(output_folder):
        raise FileExistsError(f"{output_folder} esiste già !!!")
    os.makedirs(output_folder, exist_ok=True)
    base_formatter = logging.Formatter("%(asctime)s   %(message)s", "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger("")
    logger.setLevel(logging.DEBUG)

    if info_filename != None:
        info_file_handler = logging.FileHandler(f"{output_folder}/{info_filename}")
        info_file_handler.setLevel(logging.INFO)
        info_file_handler.setFormatter(base_formatter)
        logger.addHandler(info_file_handler)

    if debug_filename != None:
        debug_file_handler = logging.FileHandler(f"{output_folder}/{debug_filename}")
        debug_file_handler.setLevel(logging.DEBUG)
        debug_file_handler.setFormatter(base_formatter)
        logger.addHandler(debug_file_handler)

    if console != None:
        console_handler = logging.StreamHandler()
        if console == "debug": console_handler.setLevel(logging.DEBUG)
        if console == "info":  console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(base_formatter)
        logger.addHandler(console_handler)

    def my_handler(type_, value, tb):
        logger.info("\n" + "".join(traceback.format_exception(type, value, tb)))

    sys.excepthook = my_handler

