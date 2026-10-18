"""Supply storage for legacy g2o constants referenced by modern pybind11."""
from pathlib import Path
import shutil
import subprocess

root = Path('/opt/native/g2opy')
source = root / 'g2o/core/hyper_graph.cpp'
text = source.read_text()
marker = 'namespace g2o {'
assert marker in text
definitions = '\n  const int HyperGraph::UnassignedId;\n  const int HyperGraph::InvalidId;\n'
if 'const int HyperGraph::UnassignedId;' not in text:
    source.write_text(text.replace(marker, marker + definitions, 1))
subprocess.run(['cmake', '--build', str(root / 'build'), '--target', 'g2o', '-j', '1'],
               check=True)
for library in (root / 'lib').glob('g2o*.so'):
    shutil.copy2(library, '/usr/local/lib/python3.10/dist-packages')
