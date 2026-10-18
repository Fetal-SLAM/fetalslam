"""Compatibility patches for the pinned upstream native Python bindings."""
from pathlib import Path

for project, external in (('g2opy', 'EXTERNAL'), ('pangolin', 'external')):
    cmake = Path('/opt/native')/project/'python/CMakeLists.txt'
    text = cmake.read_text()
    assert 'include(pybind11Tools)' in text
    text = text.replace('find_package(pybind11)', '')
    text = text.replace('include(pybind11Tools)',
                        f'add_subdirectory(${{PROJECT_SOURCE_DIR}}/{external}/pybind11 pybind11-build)')
    cmake.write_text(text)

root = Path('/opt/native/g2opy')
path = root/'python/core/eigen_types.h'
text = path.read_text()
for name in 'xyzw':
    old = f'.def("{name}", (double (Eigen::Quaterniond::*) () const) &Eigen::Quaterniond::{name})'
    assert old in text
    text = text.replace(old, f'.def("{name}", [](const Eigen::Quaterniond& q) {{ return q.{name}(); }})')
path.write_text(text)
path = root/'python/types/sba/types_six_dof_expmap.h'
text = path.read_text()
for name in ('EdgeSE3ProjectXYZ', 'EdgeSE3ProjectXYZOnlyPose', 'EdgeStereoSE3ProjectXYZ', 'EdgeStereoSE3ProjectXYZOnlyPose'):
    marker = f'.def("cam_project", &{name}::cam_project)'
    assert marker in text
    fields = ['fx', 'fy', 'cx', 'cy']
    if 'Stereo' in name:
        fields.append('bf')
    if name.endswith('OnlyPose'):
        fields.append('Xw')
    text = text.replace(marker, marker + ''.join(f'\n        .def_readwrite("{field}", &{name}::{field})' for field in fields))
path.write_text(text)
