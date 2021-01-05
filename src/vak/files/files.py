import fnmatch
from pathlib import Path
import re

from ..config.converters import expanded_user_path_str
from ..validation import is_a_directory


def find_fname(fname, ext):
    """given a file extension, finds a filename with that extension within
    another filename. Useful to find e.g. names of audio files in
    names of spectrogram files.

    Parameters
    ----------
    fname : str
        filename to search for another filename with a specific extension
    ext : str
        extension to search for in filename

    Returns
    -------
    sub_fname : str or None

    Examples
    --------
    >>> sub_fname(fname='llb3_0003_2018_04_23_14_18_54.wav.mat', ext='wav')
    'llb3_0003_2018_04_23_14_18_54.wav'
    """
    m = re.match(f'[\S]*{ext}', fname)
    if hasattr(m, 'group'):
        return m.group()
    elif m is None:
        return m


def from_dir(dir_path, ext):
    """gets all files with a given extension
    from a directory or its sub-directories.

    Parameters
    ----------
    dir_path : str, Path
        path to target directory.
        If a string, and the string ends with '/**', then
        this function will recursively search all sub-directories
        for the specified file extension.
    ext : str
        file extension to search for. E.g., 'wav' or 'npz'

    Returns
    -------
    files : list
        of paths to files with specified file extension

    Notes
    -----
    ``from_dir`` is case insensitive. For example, if you specify the extension
    as ``wav`` then it will return files that end in ``.wav`` or ``.WAV``.
    Similarly, if you specify ``TextGrid`` as the extension,
    it will return files that end in ``.textgrid`` and ``.TextGrid``.

    If no files with the specified extension are found in the directory,
    then the function looks in all directories within ``dir_path``
    and returns any files with the extension in those directories.
    Currently the function does not recurse, i.e., it does not look
    any deeper than one level below ``dir_path``.

    used by vak.io.audio.files_from_dir and vak.io.annot.files_from_dir
    """
    dir_path = str(dir_path)
    dir_path = expanded_user_path_str(dir_path)
    is_a_directory(dir_path)

    # use fnmatch + re to make search case-insensitive
    # adopted from:
    # https://gist.github.com/techtonik/5694830
    # https://jdhao.github.io/2019/06/24/python_glob_case_sensitivity/
    glob_pat = f'*.{ext}'
    rule = re.compile(fnmatch.translate(glob_pat), re.IGNORECASE)

    if dir_path.endswith('/**'):
        dir_path = Path(dir_path[:-2])
        glob_pat = f'**/{glob_pat}'
    else:
        dir_path = Path(dir_path)

    files = sorted(dir_path.glob(glob_pat))
    files = [
        file
        for file in files
        if file.is_file() and rule.match(file.name)
    ]

    if len(files) == 0:
        raise FileNotFoundError(
            f'No files with extension {ext} found in '
            f'{dir_path} or immediate sub-directories'
        )

    # TODO: use / return Path instead of strings
    return [str(file) for file in files]
