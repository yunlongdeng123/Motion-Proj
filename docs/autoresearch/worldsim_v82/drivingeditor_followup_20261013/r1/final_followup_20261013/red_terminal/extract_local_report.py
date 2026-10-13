"""Extract one owned saved-report archive to a new outputs directory."""
import argparse
from pathlib import Path, PurePosixPath
import shutil
import tarfile


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
output = args.output.resolve()
assert output.parent.name == 'outputs' and output.name == 'drivingeditor_red_GN11_30k'
assert not output.exists(), 'Refuse to overwrite existing report directory'
prefix = 'red_gn11_30k_saved_cpu_20261013'
with tarfile.open(args.archive, 'r:gz') as archive:
    members = archive.getmembers()
    safe = []
    for member in members:
        path = PurePosixPath(member.name)
        assert path.parts[0] == prefix and '..' not in path.parts and not path.is_absolute()
        assert member.isdir() or member.isfile(), 'Refuse archive links or special files'
        destination = output.joinpath(*path.parts[1:]).resolve()
        assert destination == output or output in destination.parents
        safe.append((member, destination))
    output.mkdir()
    for member, destination in safe:
        if member.isdir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.extractfile(member) as source, destination.open('xb') as target:
            shutil.copyfileobj(source, target)
print('Extracted actual saved report to new directory: ' + str(output))
