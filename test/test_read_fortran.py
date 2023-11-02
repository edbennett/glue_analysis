#!/usr/bin/env python3

from io import StringIO

from glue_analysis.readers.read_fortran import _read_correlators_fortran


def test_read_correlators_fortran_records_filename() -> None:
    filename = "testname.txt"
    corr_file = StringIO("column-name")
    answer = _read_correlators_fortran(corr_file, filename)
    assert answer.filename == filename
