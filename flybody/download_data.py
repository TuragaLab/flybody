"""Script for downloading supplementary flybody data from figshare:
https://doi.org/10.25378/janelia.25309105
"""

import os
import requests
import zipfile


def figshare_download(keys: str | list[str]):
    urls = {
        'controller-reuse-checkpoints':
            'https://janelia.figshare.com/ndownloader/files/51196886',
        'walking-imitation-dataset':
            'https://janelia.figshare.com/ndownloader/files/51196868',
        'flight-imitation-dataset':
            'https://janelia.figshare.com/ndownloader/files/51196859',
        'trained-policies':
            'https://janelia.figshare.com/ndownloader/files/44815195',
    }
    if isinstance(keys, str):
        keys = [keys]
    for key in keys:
        assert key in urls, ('Invalid key. Only the following keys are supported: ' +
                            ', '.join(urls.keys()))
        download_and_unzip(urls[key])


def download_and_unzip(
    url: str,
    dest_path: str = 'flybody-data',
    delete_zip: bool = True
):
    """Download file from url to `dest_path` and unzip it."""
    # Download.
    response = requests.get(url)
    assert response.ok
    zip_fname = response.headers['Content-Disposition'].split('filename=')[1]
    zip_fname = zip_fname.replace('"', '')
    os.makedirs(dest_path, exist_ok=True)
    full = os.path.join(dest_path, zip_fname)
    with open(full, 'wb') as f:
        f.write(response.content)
    
    # Unzip.
    with zipfile.ZipFile(full, 'r') as zip_ref:
        directory = zip_fname[:-4]  # Remove ".zip"
        zip_ref.extractall(os.path.join(dest_path, directory))
    
    if delete_zip:
        # Delete zip file.
        os.remove(full)
