# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from libc.stdlib cimport malloc, free
from libc.string cimport strlen

cdef inline int min3(int a, int b, int c) nogil:
    """Fast minimum of three integers."""
    if a <= b:
        if a <= c:
            return a
        else:
            return c
    else:
        if b <= c:
            return b
        else:
            return c


def levenshtein(str s1, str s2):
    """Fast Cython implementation of Levenshtein distance.
    
    Parameters
    ----------
    s1, s2 : str
        Input strings to compare
        
    Returns
    -------
    int
        Levenshtein distance between s1 and s2
    """
    cdef:
        int len1 = len(s1)
        int len2 = len(s2)
        int i, j
        int cost
        int *prev
        int *curr
        int *temp
        int result
        
    # Quick returns
    if s1 == s2:
        return 0
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1
        
    # Ensure s1 is the longer string
    if len1 < len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1
    
    # Allocate arrays
    prev = <int*>malloc((len2 + 1) * sizeof(int))
    curr = <int*>malloc((len2 + 1) * sizeof(int))
    
    if not prev or not curr:
        if prev:
            free(prev)
        if curr:
            free(curr)
        raise MemoryError("Failed to allocate memory for Levenshtein calculation")
    
    # Initialize first row
    for j in range(len2 + 1):
        prev[j] = j
    
    # Main computation
    for i in range(len1):
        curr[0] = i + 1
        
        for j in range(len2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
                
            curr[j + 1] = min3(
                prev[j + 1] + 1,    # deletion
                curr[j] + 1,        # insertion
                prev[j] + cost      # substitution
            )
        
        # Swap arrays
        temp = prev
        prev = curr
        curr = temp
    
    result = prev[len2]
    
    # Clean up
    free(prev)
    free(curr)
    
    return result


def levenshtein_bounded(str s1, str s2, int max_dist):
    """Fast Cython implementation of bounded Levenshtein distance.
    
    Returns early if distance exceeds max_dist.
    
    Parameters
    ----------
    s1, s2 : str
        Input strings to compare
    max_dist : int
        Maximum distance to compute
        
    Returns
    -------
    int
        Levenshtein distance between s1 and s2, or max_dist + 1 if exceeded
    """
    cdef:
        int len1 = len(s1)
        int len2 = len(s2)
        int i, j
        int cost
        int *prev
        int *curr
        int *temp
        int result
        int min_dist
        
    # Quick returns
    if s1 == s2:
        return 0
    if abs(len1 - len2) > max_dist:
        return max_dist + 1
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1
        
    # Ensure s1 is the longer string
    if len1 < len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1
    
    # Allocate arrays
    prev = <int*>malloc((len2 + 1) * sizeof(int))
    curr = <int*>malloc((len2 + 1) * sizeof(int))
    
    if not prev or not curr:
        if prev:
            free(prev)
        if curr:
            free(curr)
        raise MemoryError("Failed to allocate memory for Levenshtein calculation")
    
    # Initialize first row
    for j in range(len2 + 1):
        prev[j] = j
    
    # Main computation with early termination
    for i in range(len1):
        curr[0] = i + 1
        min_dist = max_dist + 1
        
        for j in range(len2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
                
            curr[j + 1] = min3(
                prev[j + 1] + 1,    # deletion
                curr[j] + 1,        # insertion
                prev[j] + cost      # substitution
            )
            
            if curr[j + 1] < min_dist:
                min_dist = curr[j + 1]
        
        # Early termination if minimum distance in row exceeds threshold
        if min_dist > max_dist:
            free(prev)
            free(curr)
            return max_dist + 1
        
        # Swap arrays
        temp = prev
        prev = curr
        curr = temp
    
    result = prev[len2]
    
    # Clean up
    free(prev)
    free(curr)
    
    return result