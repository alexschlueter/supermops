from pathlib import Path
from tqdm import tqdm
import zipfile

DATA_URL = "https://data-management.uni-muenster.de/direct-access/wwurdm/09978425560"
EXPECTED_SIZE = 197737035
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DOWNLOAD_PATH = REPO_ROOT / "article/data/article_data.zip"
DATA_EXTRACT_PATH = REPO_ROOT / "article/data"

# https://stackoverflow.com/a/63831344
def download(url, filename, desc="", file_size=None):
    import functools
    import pathlib
    import shutil
    import requests
    
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")

    file_size = int(r.headers.get('Content-Length', file_size))

    path = pathlib.Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)

    return path

def prepare_article_data():
    print(f"Target path: {DATA_EXTRACT_PATH}")
    download(DATA_URL, DATA_DOWNLOAD_PATH, desc="Downloading...", file_size=EXPECTED_SIZE)
    with zipfile.ZipFile(DATA_DOWNLOAD_PATH) as parent_zip:
        for subfolder in ["ground_truth.zip", "evaluation_results.zip"]:
            with zipfile.ZipFile(parent_zip.open(subfolder)) as subzip:
                for f in tqdm(subzip.infolist(), desc=f"Extracting {subfolder}..."):
                    subzip.extract(f, DATA_EXTRACT_PATH)
    print("Removing parent zip...")
    DATA_DOWNLOAD_PATH.unlink()
                
if __name__ == "__main__":
    prepare_article_data()