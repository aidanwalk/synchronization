"""
Functions to assist with handing the commonly used data/files in this pipeline.


FUNCTIONS:
----------
find_object_files: 
    Find all files in a directory that match a given object name.
    Can be used for both FITS files and _cal.npz files.
    Options to filter by observation time and count total frames.
find_files_in_dir: 
    Find all files in a directory with a given file type, 
    optionally filtering by observation time.
keep_files_between_times:
    Filter a list of files to keep only those within a specified time range.
find_file_time:
    Extract the observation time from a filename using regex.
open_PSF_params:
    Load PSF parameters from a .npz file or list of .npz files.




"""

import os 
import re
import sys
from tqdm import tqdm
import numpy as np
from astropy.io import fits
import warnings




def load_dark_frame(filepath, dir='') -> np.ndarray:
    """
    Loads the dark frame. If the dark frame is a datacube, 
    it will be median combined across axis 0.
    
    Parameters:
    -----------
    filepath : str
        Path to the dark frame file. Can be a single image or a datacube.
        
    Returns:
    --------
    numpy.ndarray
        The dark frame to subtract from each data frame before averaging.
        If the dark frame does not exist, returns None.
        
    """
    dark_frame = None
    
    try:
        print(f'Loading dark frame from {filepath} ...')
        dark_frame = fits.getdata(filepath)
        
        # If the dark frame is a datacube, median combine it across axis 0
        if dark_frame.ndim == 3:
            dark_frame = np.median(dark_frame, axis=0)
        
            
    except FileNotFoundError: 
        warnings.warn(f'Dark frame file {filepath} does not exist. No dark subtraction will be performed.')
        
        
    return dark_frame




def find_object_files(
    directory, 
    object_name=None, 
    file_type='.fits', 
    obs_start=None, 
    obs_end=None, 
    count_frames=False
    ) -> list[str]:
    """
    Find all data files in the specified directory that match the given 
    object name.
    
    File extensions are stripped from the filenames. 
    
    Parameters:
    -----------
    directory : str
        Path to the directory containing data files.
    object_name : str
        The object name to search for in the FITS headers or npz file. 
        header keyword 'OBJECT'.
        if None, returns all files with extension <file_type>
    file_type : str
        File extension you want to search in (and contains the object name).
        Can be .fits or _cal.npz
    obs_start : str
        Observation start time in format 'HH:MM:SS' (Really any time format 
        consistent with the filenames will work).
        If provided, only files observed after this time will be returned.
    obs_end : str
        Observation end time in format 'HH:MM:SS'.
        If provided, only files observed before this time will be returned.
    count_frames : bool
        If True, also return the total number of frames across all datacubes.
        
    Returns:
    --------
    filenames : list
        List of filenames that match the specified object name.
    """
    
    files = find_files_in_dir(directory, file_type=file_type, 
                              obs_start=obs_start, obs_end=obs_end)
    
    if object_name is None: return files
    
    # If we actually want to filter by object name, 
    assert file_type in ['.fits', '_cal.npz'], "file_type must be .fits or _cal.npz"
    
    
    object_files = []
    N_frames = 0
    
    for filename in tqdm(files):
        filepath = os.path.join(directory, filename)
        if file_type == '.fits':
            hdr = fits.getheader(filepath)
        elif file_type == '_cal.npz':
            hdr = np.load(filepath, allow_pickle=True)['header'].item(0)
        else:
            sys.exit('File type not supported. Cannot extract object information.')
            
        # If this is the object we are looking for
        if hdr['OBJECT'] == object_name:
            # include it in the returned data
            object_files.append(filename)
            N_frames += hdr['NAXIS3']

    if not count_frames:
        return sorted(object_files)
    else:
        return sorted(object_files), N_frames





def find_files_in_dir(
    directory,
    file_type='.fits',
    obs_start=None,
    obs_end=None
    ) -> list[str]:
    """
    Find all data files in the specified directory with the given file type.
    
    Parameters:
    -----------
    directory : str
        Path to the directory containing data files.
    file_type : str
        File extension you want to search in (e.g. .fits or _cal.npz)
        
    Returns:
    --------
    filenames : list
        List of filenames that match the specified file type.
    """
    
    files = sorted(os.listdir(directory))
    matched_files = [f for f in files if f.endswith(file_type)]
    
    if obs_start is not None and obs_end is not None:
        print('Filtering files between {} and {}.'.format(obs_start, obs_end))
        matched_files = keep_files_between_times(matched_files, obs_start, obs_end)
    
    return sorted(matched_files)





def keep_files_between_times(files, start_time, end_time, PLred_time_format=False):
    """
    Keep only files whose observation times are within the given range.
    
    Parameters:
    -----------
    files : list
        List of data filenames to process.
    start_time : str
        Observation start time in format 'HH:MM:SS'
    end_time : str
        Observation end time in format 'HH:MM:SS'
    PLred_time_format : bool
        If True, the function will look for times in the format 'T15h14m15s' 
        which is used in the PLred pipeline output files. 
        If False, it will look for times in the format '15:14:15' which is used 
        in the original datacube filenames.
        
    Returns:
    --------
    kept_files : list
        List of files within the given time range.
    """
    
    if PLred_time_format:
        # Convert start_time and end_time to PLred format 'T15h14m15s'
        start_time = 'T' + start_time.replace(':', 'h', 1).replace(':', 'm', 1).replace('.', 's', 1)
        end_time = 'T' + end_time.replace(':', 'h', 1).replace(':', 'm', 1).replace('.', 's', 1)
        # Get rid of any decimal seconds for matching, since some files have different precision in the timestamp
        start_time = re.sub(r's\d+s', 's', start_time)
        end_time = re.sub(r's\d+s', 's', end_time)
        pattern = r'T\d{2}h\d{2}m\d{2}s'  # Matches 'T15h14m15s'
        regex = re.compile(pattern)
    
    else:
        # Create a regex pattern from the start_time format
        pattern = r'(?<!\d)' + re.sub(r'\d', r'\\d', start_time) + r'(?!\d)'  
    
    
    kept_files = []
    # handle the case where file systems can have either ':' or '_' as the separator
    if any(':' in f for f in files):
        pass # do nothing, the pattern is already correct
    else:
        pattern = pattern.replace(':', '_') # Matches 'YYYYMMDD_HH_MM_SS'
    # print(f'Using time pattern: {pattern}')
    regex = re.compile(pattern)
    
    for i, filename in enumerate(files):
        next_file = files[i+1] if i < len(files)-1 else None
        file_time = find_file_time(filename, regex=regex)
        next_file_time = find_file_time(next_file, regex=regex) if next_file is not None else file_time
        
        if file_time is None:
            continue
        
        
        file_time = file_time.replace('_', ':')  if ':' not in file_time else file_time
        if (start_time <= file_time <= end_time) or \
          (file_time <= start_time and next_file_time >= start_time):
            kept_files.append(filename)

    return kept_files




def match_time_to_PLred_files(files, time):
    """
    Find the file in the list that matches the given time
    
    Parameters:
    -----------
    files : list
        List of data filenames to search through. These are the files produced
        by the PLred pipeline, so their time format is something like 
        'firstpl_2026-02-27T15h14m15s_BETLIB_P.fits'
    time : str
        The time to match in format 'HH:MM:SS'
        
    
    Returns:
    --------
    matched_file : str
        The filename that matches the given time. If no match is found, 
        returns None.
    """
    
    pattern = r'T\d{2}h\d{2}m\d{2}s'  # Matches 'T15h14m15s'
    regex = re.compile(pattern)
    
    matched_files = []
    for filename in files:
        file_time = find_file_time(filename, regex=regex)
        if file_time is None:
            continue
        
        # Convert file_time from 'T15h14m15s' to '15:14:15' for comparison
        file_time = file_time[1:].replace('h', ':').replace('m', ':').replace('s', '')
        if file_time == time:
            matched_files.append(filename)
    
    if len(matched_files) == 0:
        # print(f'No files found matching time {time}.')
        return None
    elif len(matched_files) > 1:
        print(f'Multiple files found matching time {time}: {matched_files}. Returning the first match.')
        
    return matched_files[0]




def find_file_time(filename, pattern=r'(?<!\d)\d\d:\d\d:\d\d(?!\d)', regex=None):
    """
    Extract the time from a filename using a regex pattern.
    
    Parameters:
    -----------
    filename : str
        The filename from which to extract the time.
    pattern : str
        The regex pattern to use for extracting the time.
        Default pattern matches HH:MM:SS.ssssss format.
        
    Returns:
    --------
    time_str : str
        The extracted time string.
    """
        
    if regex is None:
        pattern = pattern if ':' in filename else pattern.replace(':', '_') 
        match = re.search(pattern, filename)
    else:
        match = regex.search(filename)
    if match:
        return match.group(0)
    else:
        if regex is not None: pattern = regex.pattern
        print(f'No time found in filename {filename} using pattern {pattern}.')
        return None




def open_PSF_params(filepath, dir=''):
    """
    Open a PSF parameters file and return the parameters.
    
    Parameters:
    -----------
    filepath : str or list of str
        Path to the PSF parameters .npz file.
        If a list of filepaths is given, the function will return a list of 
        PSF parameter arrays.
    dir : str
        Directory where the file is located. The default is the current
        working directory.
        
    Returns:
    --------
    PSF_params : np.ndarray
        Array of PSF parameters with shape (N_frames, 5)
        Columns: [x_psf, y_psf, x_FWHM, y_FWHM, strehl]
    """
    if isinstance(filepath, str):
        PSF_params = np.load(os.path.join(dir, filepath))
        return PSF_params
    
    elif isinstance(filepath, list):
        PSF_params = []
        for fp in filepath:
            data = np.load(os.path.join(dir, fp))
            PSF_params.append(data)
        return PSF_params
    
    else: 
        sys.exit('filepath must be a string or list of strings.')
        
    
    
    
def read_telemetry(filepath):
    """
    Get the telemetry data for each frame in the telemetry data file
    
    Parameters:
    -----------
    filepath : str
        Path to the telemetry file.
        
    Returns:
    --------
    telemetry : list of lists
        A list containing the relevent telemetry data for every frame. 
        telemetry shape = (N_frames, 3)
        row = [frame_index, frame_start, frame_end]
    """
    
    # Open the telemetry data
    data = np.genfromtxt(filepath)
    
    frame_telemetry = []
    for i, row in enumerate(data):
        frame_idx = int(row[0])
        t_end = row[4]
        # Assume the frame began when the last one ended
        if i == 0:
            t_start = np.inf  # Very large positive number
        else:
            prev_row = data[i-1]
            t_start = prev_row[4]
        
        frame_telemetry.append([frame_idx, t_start, t_end])
    
    
    # We can't solve for t_start of the first frame since there is no previous
    # entry. 
    # frame_telemetry.pop(0)
    
    return frame_telemetry