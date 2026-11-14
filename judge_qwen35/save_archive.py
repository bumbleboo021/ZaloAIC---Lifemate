"""Write the gzip-compressed Docker image without a second uncompressed archive."""
import gzip
from pathlib import Path
import shutil
import subprocess
import sys

image, path = sys.argv[1:]
archive = Path(path)
pending = archive.with_suffix(".gz.partial")
with pending.open("xb") as raw:
    with subprocess.Popen(["docker", "--context", "desktop-linux", "save", image],
                          stdout=subprocess.PIPE) as proc:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=3, mtime=0) as output:
            shutil.copyfileobj(proc.stdout, output, length=8 * 1024 * 1024)
        if proc.wait():
            raise RuntimeError("docker save failed")
pending.replace(archive)
