from .read_binary import read_correlators_binary
from .read_fortran import read_correlators_fortran

readers = {
    "binary": read_correlators_binary,
    "fortran": read_correlators_fortran,
}
