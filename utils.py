import random
import torch
import numpy as np
import yaml
import os
import requests
import secrets
from typing import Any, Generator, Iterable, List, Mapping, Optional, Sequence, Sized, Union, Collection
from pathlib import Path
from urllib.parse import urlencode, parse_qs, urlsplit, urlunsplit, urlparse
from logging import getLogger
import tarfile
import zipfile
from hashlib import md5
import gzip
from tqdm import tqdm

log = getLogger(__name__)

def set_global_seeds(i):
    torch.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    random.seed(i)
    np.random.seed(i)


def get_config(path):
    with open(path, 'r') as stream:
        config = yaml.load(stream)
    return config

def get_version(filename):
    n = filename.count('_')
    if n == 0:
        return filename
    elif n==1:
        return filename.split('_')[-1]
    elif n==2:
        return filename.split('_')[-2]
    else:
        print('Cant find version')

def get_last_save(path):
    if os.path.exists(path):
        files = [get_version('.'.join(f.split('.')[:-1])) for f in os.listdir(path) if '.pt' in f]
        numbers = []
        for f in files:
            try:
                numbers.append(int(f))
            except: pass
        if len(numbers) > 0:
            return max(numbers)
        else:
            return 0
    else:
        return 0


def simple_download(url: str, destination: Union[Path, str]) -> None:
    """Download a file from URL to target location.
    Displays a progress bar to the terminal during the download process.
    Args:
        url: The source URL.
        destination: Path to the file destination (including file name).
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    log.info('Downloading from {} to {}'.format(url, destination))

    if url.startswith('s3://'):
        return s3_download(url, str(destination))

    chunk_size = 32 * 1024

    temporary = destination.with_suffix(destination.suffix + '.part')

    headers = {'dp-token': _get_download_token()}
    r = requests.get(url, stream=True, headers=headers)
    if r.status_code != 200:
        raise RuntimeError(f'Got status code {r.status_code} when trying to download {url}')
    total_length = int(r.headers.get('content-length', 0))

    if temporary.exists() and temporary.stat().st_size > total_length:
        temporary.write_bytes(b'')  # clearing temporary file when total_length is inconsistent

    with temporary.open('ab') as f:
        done = False
        downloaded = f.tell()
        if downloaded != 0:
            log.warning(f'Found a partial download {temporary}')
        with tqdm(initial=downloaded, total=total_length, unit='B', unit_scale=True) as pbar:
            while not done:
                if downloaded != 0:
                    log.warning(f'Download stopped abruptly, trying to resume from {downloaded} '
                                f'to reach {total_length}')
                    headers['Range'] = f'bytes={downloaded}-'
                    r = requests.get(url, headers=headers, stream=True)
                    if 'content-length' not in r.headers or \
                            total_length - downloaded != int(r.headers['content-length']):
                        raise RuntimeError(f'It looks like the server does not support resuming '
                                           f'downloads.')
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:  # filter out keep-alive new chunks
                        downloaded += len(chunk)
                        pbar.update(len(chunk))
                        f.write(chunk)
                if downloaded >= total_length:
                    # Note that total_length is 0 if the server didn't return the content length,
                    # in this case we perform just one iteration and assume that we are done.
                    done = True

    temporary.rename(destination)


def download(dest_file_path: [List[Union[str, Path]]], source_url: str, force_download: bool = True) -> None:
    """Download a file from URL to one or several target locations.
    Args:
        dest_file_path: Path or list of paths to the file destination (including file name).
        source_url: The source URL.
        force_download: Download file if it already exists, or not.
    """

    if isinstance(dest_file_path, list):
        dest_file_paths = [Path(path) for path in dest_file_path]
    else:
        dest_file_paths = [Path(dest_file_path).absolute()]

    if not force_download:
        to_check = list(dest_file_paths)
        dest_file_paths = []
        for p in to_check:
            if p.exists():
                log.info(f'File already exists in {p}')
            else:
                dest_file_paths.append(p)

    if dest_file_paths:
        cache_dir = os.getenv('DP_CACHE_DIR')
        cached_exists = False
        if cache_dir:
            first_dest_path = Path(cache_dir) / md5(source_url.encode('utf8')).hexdigest()[:15]
            cached_exists = first_dest_path.exists()
        else:
            first_dest_path = dest_file_paths.pop()

        if not cached_exists:
            first_dest_path.parent.mkdir(parents=True, exist_ok=True)

            simple_download(source_url, first_dest_path)
        else:
            log.info(f'Found cached {source_url} in {first_dest_path}')

        for dest_path in dest_file_paths:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(first_dest_path), str(dest_path))


def download_decompress(url: str,
                        download_path: Union[Path, str],
                        extract_paths: Optional[Union[List[Union[Path, str]], Path, str]] = None) -> None:
    """Download and extract .tar.gz or .gz file to one or several target locations.
    The archive is deleted if extraction was successful.
    Args:
        url: URL for file downloading.
        download_path: Path to the directory where downloaded file will be stored until the end of extraction.
        extract_paths: Path or list of paths where contents of archive will be extracted.
    """
    file_name = Path(urlparse(url).path).name
    download_path = Path(download_path)

    if extract_paths is None:
        extract_paths = [download_path]
    elif isinstance(extract_paths, list):
        extract_paths = [Path(path) for path in extract_paths]
    else:
        extract_paths = [Path(extract_paths)]

    cache_dir = os.getenv('DP_CACHE_DIR')
    extracted = False
    if cache_dir:
        cache_dir = Path(cache_dir)
        url_hash = md5(url.encode('utf8')).hexdigest()[:15]
        arch_file_path = cache_dir / url_hash
        extracted_path = cache_dir / (url_hash + '_extracted')
        extracted = extracted_path.exists()
        if not extracted and not arch_file_path.exists():
            simple_download(url, arch_file_path)
        else:
            if extracted:
                log.info(f'Found cached and extracted {url} in {extracted_path}')
            else:
                log.info(f'Found cached {url} in {arch_file_path}')
    else:
        arch_file_path = download_path / file_name
        simple_download(url, arch_file_path)
        extracted_path = extract_paths.pop()

    if not extracted:
        log.info('Extracting {} archive into {}'.format(arch_file_path, extracted_path))
        extracted_path.mkdir(parents=True, exist_ok=True)

        if file_name.endswith('.tar.gz'):
            untar(arch_file_path, extracted_path)
        elif file_name.endswith('.gz'):
            ungzip(arch_file_path, extracted_path / Path(file_name).with_suffix('').name)
        elif file_name.endswith('.zip'):
            with zipfile.ZipFile(arch_file_path, 'r') as zip_ref:
                zip_ref.extractall(extracted_path)
        else:
            raise RuntimeError(f'Trying to extract an unknown type of archive {file_name}')

        if not cache_dir:
            arch_file_path.unlink()

    for extract_path in extract_paths:
        for src in extracted_path.iterdir():
            dest = extract_path / src.name
            if src.is_dir():
                _copytree(src, dest)
            else:
                extract_path.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(src), str(dest))




def _get_download_token() -> str:
    """Return a download token from ~/.deeppavlov/token file.
    If token file does not exists, creates the file and writes to it a random URL-safe text string
    containing 32 random bytes.
    Returns:
        32 byte URL-safe text string from ~/.deeppavlov/token.
    """
    token_file = Path.home() / '.deeppavlov' / 'token'
    if not token_file.exists():
        if token_file.parent.is_file():
            token_file.parent.unlink()
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(secrets.token_urlsafe(32), encoding='utf8')

    return token_file.read_text(encoding='utf8').strip()



def untar(file_path: Union[Path, str], extract_folder: Optional[Union[Path, str]] = None) -> None:
    """Simple tar archive extractor.
    Args:
        file_path: Path to the tar file to be extracted.
        extract_folder: Folder to which the files will be extracted.
    """
    file_path = Path(file_path)
    if extract_folder is None:
        extract_folder = file_path.parent
    extract_folder = Path(extract_folder)
    tar = tarfile.open(file_path)
    tar.extractall(extract_folder)
    tar.close()


def ungzip(file_path: Union[Path, str], extract_path: Optional[Union[Path, str]] = None) -> None:
    """Simple .gz archive extractor.
    Args:
        file_path: Path to the gzip file to be extracted.
        extract_path: Path where the file will be extracted.
    """
    chunk_size = 16 * 1024
    file_path = Path(file_path)
    if extract_path is None:
        extract_path = file_path.with_suffix('')
    extract_path = Path(extract_path)

    with gzip.open(file_path, 'rb') as fin, extract_path.open('wb') as fout:
        while True:
            block = fin.read(chunk_size)
            if not block:
                break
            fout.write(block)
