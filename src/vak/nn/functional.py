"""functional interface to ``vak``-specific tensor operations"""
import torch


__all__ = [
    'levenshtein',
    'lbl_tb2labels',
    'remove_unlabeled',
    'segment_error_rate',
]


def levenshtein(source, target):
    """Levenshtein distance: number of deletions, insertions,
    or substitutions required to convert source string
    into target string.

    Parameters
    ----------
    source : torch.Tensor

    target : torch.Tensor

    Returns
    -------
    distance : int
        number of deletions, insertions, or substitutions
        required to convert source into target.

    adapted from https://github.com/toastdriven/pylev/blob/master/pylev.py
    to fix issues with the Numpy implementation in
    https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python

    >>> source = torch.tensor([ord(char) for char in 'kitten'])
    >>> print(source)
    tensor([107, 105, 116, 116, 101, 110])
    >>> target = torch.tensor([ord(char) for char in 'sitting'])
    >>> print(target)
    tensor([115, 105, 116, 116, 105, 110, 103])
    >>> levenshtein(source, target)
    tensor(3)
    """
    if torch.equal(source, target):
        return torch.tensor(0)

    len_source = torch.numel(source)
    len_target = torch.numel(target)

    if len_source == 0:
        return torch.tensor(len_target)
    if len_target == 0:
        return torch.tensor(len_source)

    if len_source > len_target:
        source, target = target, source
        len_source, len_target = len_target, len_source

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    d0 = torch.arange(len_target + 1)
    d1 = torch.arange(len_target + 1)
    for i in range(len_source):
        d1[0] = i + 1
        for j in range(len_target):
            cost = d0[j]

            if source[i] != target[j]:
                cost += 1  # substitution

                x_cost = d1[j] + 1  # insertion
                if x_cost < cost:
                    cost = x_cost

                y_cost = d0[j + 1] + 1
                if y_cost < cost:
                    cost = y_cost

            d1[j + 1] = cost

        d0, d1 = d1, d0

    return d0[-1]


def segment_error_rate(source, target):
    """Levenshtein edit distance between two tensors
    (source, target) normalized by the length of the
    target sequence, which is assumed to be the ground truth.
    Also known as word error distance; here applied to other vocalizations
    in addition to speech.

    Parameters
    ----------
    source : torch.Tensor
    target : torch.Tensor

    Returns
    -------
    levenshtein(source, target) / torch.numel(target)

    Examples
    --------
    >>> source = torch.tensor([ord(char) for char in 'kitten'])
    >>> print(source)
    tensor([107, 105, 116, 116, 101, 110])
    >>> target = torch.tensor([ord(char) for char in 'sitting'])
    >>> print(target)
    tensor([115, 105, 116, 116, 105, 110, 103])
    >>> levenshtein(source, target)
    tensor(3)
    """
    return levenshtein(source, target) / torch.numel(target)


def remove_unlabeled(y, unlabeled=0):
    """remove "unlabeled" / "silent gap" labels
    from a tensor representing labeled time bins.

    Used for segmenting, i.e.,
    converting a series of labeled frames,
    including an "unlabeled" or "silent gap" class,
    into a sequence of segment labels

    Parameters
    ----------
    y : torch.tensor
        containing integers representing labels
    unlabeled : int
        integer that represents "unlabeled" / "silent gap" class label.
        Default is 0.

    Examples
    --------
    >>> y = torch.tensor([0, 0, 0, 1, 1, 1, 1, 0, 0, 2, 2, 0, 0])
    >>> remove_unlabeled(y, unlabeled=0)
    tensor([1, 1, 1, 1, 2, 2])
    """
    unlabeled = torch.tensor(unlabeled).to(y.device)
    return y[y != unlabeled]


def lbl_tb2labels(y, unlabeled=0):
    """converts output of network from label for each frame
    to one label for each continuous segment.
    Removes "unlabeled" / "silent gap"

    Pure ``torch`` implementation of ``vak.labeled_timebins.lbl_tb2labels``.
    Unlike that function, this version does not map integer predictions
    from a neural network back to user-defined labels such as strings.

    Parameters
    ----------
    y : torch.tensor
        containing integers representing labels
    unlabeled : int
        integer that represents "unlabeled" / "silent gap" class label.
        Default is 0.

    >>> y = torch.tensor([0, 0, 0, 1, 1, 1, 1, 0, 0, 2, 2, 0, 0])
    >>> lbl_tb2labels(y, unlabeled=0)
    tensor([1, 2])
    """
    y_labels = torch.unique_consecutive(y)
    if unlabeled is not None:
        y_labels = remove_unlabeled(y_labels, unlabeled=unlabeled)
    return y_labels
