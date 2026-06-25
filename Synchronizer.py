"""
Module to syncronize focal plane images with Photonic Lantern data. 

Functions:
--------
iter_overlapping_datacubes(reference_datacube_files, other_datacube_files):
    Generator that yields reference datacube files and their overlapping
    other datacube files based on the time in the file names.
binary_overlap(ref_start, ref_end, times):
    Determine which times in the list overlap with the reference time range.
compute_fractional_overlaps(reference_telemetry_file, other_telemetry_files, ref_dir='', other_dir='', frame_offset=0.):
    Computes the overlap percentage of individual frames in overlapping_tel_files
    to the frames in ref_tel_file
    

Classes:
--------
Overlaps(input_dict, ref_cubesize, frame_offset, *args, **kwargs):
    Class to hold overlap data. 


json Files:
-----------
A json file is made for each reference (photonic lantern) datacube file. And is 
used to communicate exactly how the focal plane images overlap with each PL 
frame. 

The json file contains a dict with structure:
    file_name = name of the reference datacube file for which overlaps are computed
    overlaps = {
        ref_frame_index : [
            overlapping_datacube_file,
            overlapping_frame_index,
            overlap_fraction
        ]
    }



@author: Aidan Walk
@email: walka@hawaii.edu
@date: 2025-12-12

"""



import os
import data_handling
import numpy as np
# from tqdm import tqdm
import json
# from astropy.io import fits
import fitsio



def iter_overlapping_datacubes(reference_datacube_files, other_datacube_files):
    """
    Find overlapping datacubes for reference_datacube_files based exclusively
    on the times in their filenames. 
    
    Example usage: 
    
    for ref_file, overlapping_cubes in Synchronizer.find_overlapping_datacubes(
        PL_telemetry_files,
        FP_telemetry_files
    ):
        print(f'Reference file: {ref_file} has {len(overlapping_cubes)} overlapping cubes.')
        
        overlaps = compute_fractional_overlaps(
            ref_file,
            overlapping_cubes
        )
    
    
    Parameters:
    -----------
    reference_datacube_files : list[str]
        List of reference datacube filenames to iterate over.
    other_datacube_files : list[str]
        List of other datacube filenames to check for overlaps.
        
    yields:
    --------
    reference_datacube_file : str
        reference datacube filename.
    overlapping_other_datacube_files : list[str]
        List of other datacube filenames that overlap with the reference 
        datacube.
    """
    # Find the times of each datacube
    reference_times = [
        data_handling.find_file_time(f) for f in reference_datacube_files
    ]
    other_times     = [
        data_handling.find_file_time(f) for f in other_datacube_files
    ]
    
    
    other_datacube_files = np.array(other_datacube_files)
    
    for i, ref_time in enumerate(reference_times):
        # Try to get the file time of the next reference datacube
        # So we can define the time range of the current datacube
        try: next_ref_time = reference_times[i+1] 
        except: next_ref_time = '99'  # Far future time
        
        overlapping_indices = binary_overlap(
            ref_start=ref_time,
            ref_end=next_ref_time,
            times=other_times
        )
        
        if not np.any(overlapping_indices): continue
        
        yield reference_datacube_files[i], \
            other_datacube_files[overlapping_indices].tolist()




def binary_overlap(ref_start, ref_end, times):
    """
    Determine which times in the list overlap with the reference time range.
    
    Parameters:
    -----------
    ref_start : str
        Start time of the reference datacube in format 'HH:MM:SS'
        (Note: this can be any string format as long as it is consistent with times)
    ref_end : str
        End time of the reference datacube in format 'HH:MM:SS'
    times : list[str]
        List of datacube times to check for overlap.
        
    returns:
    --------
    overlapping_indices : np.ndarray (bool)
        Boolean array indicating which times overlap with the reference time range.
    """
    
    overlapping_indices = np.zeros(len(times), dtype=bool)
    # Loop through all the FP times to find where they overlap the PL data
    for j, start_time in enumerate(times):
        # try to get the next file time, if not, fill it with high values
        try: end_time = times[j+1]
        except: end_time = '99'
        
        # print(start_time, end_time, ref_start, ref_end)
        if not (end_time <= ref_start or start_time >= ref_end):
            overlapping_indices[j] = True
            
    return overlapping_indices
        



def compute_fractional_overlaps(
    reference_telemetry_file,
    other_telemetry_files,
    reference_exposure_time = None,
    other_exposure_time = None,
    ref_dir='', 
    other_dir='',
    frame_offset=0., 
    overlap_threshold=0.8
    ):
    """
    Computes the overlap percentage of individual frames in other_telemetry_files
    to the frames in datacube reference_telemetry_file. This function is best used
    in conjunction with the iter_overlapping_datacubes function.
    
    Parameters:
    -----------
    reference_telemetry_file : str
        Path to the reference datacube telemetry file
    overlapping_tel_files : list[str]
        Files possibly overlapping with the reference telemetry file datacube.
    ref_dir : str
        Directory containing the reference datacube telemetry file.
    other_dir : str
        Directory containing the overlapping datacube telemetry files.
    frame_offset : float
        Time offset applied to the reference frames in seconds.
        eg. a frame_offset = 3 means other_data is 3 seconds ahead of the reference.
        
    returns: 
    --------
    overlaps : dict
        A dict containing the overlapping frame information with structure
        {ref_frame_index : [overlapping_other_file, overlapping_frame_index, overlap_percentage]}

    """
    # Open the reference telemetry data [frame_index, frame_start, frame_end]
    PL_tel_file = os.path.join(ref_dir, reference_telemetry_file)
    ref_data = data_handling.read_telemetry(PL_tel_file)
    
    
    # Compute the optimal number of other frames overlapping with the reference
    if reference_exposure_time is None or other_exposure_time is None:
        # Try to infer exposure times from the telemetry data
        reference_exposure_time = np.median([
            frame[2] - frame[1] for frame in ref_data
        ])
        
        # Open one of the other telemetry files to infer exposure time
        sample_other_file = os.path.join(other_dir, other_telemetry_files[0])
        other_data = data_handling.read_telemetry(sample_other_file)
        other_exposure_time = np.median([
            frame[2] - frame[1] for frame in other_data
        ])
        
    N_frames_needed = reference_exposure_time / other_exposure_time
    
    
    # Open the overlapping data with structure 
    # [[file_name, frame_index, frame_start, frame_end], ... ]
    fp_data = []
    for file in other_telemetry_files:
        data = data_handling.read_telemetry(os.path.join(other_dir, file))
        data = [ [file] + frame for frame in data ]
        fp_data.extend(data)
    
    
    
    # Compute the overlapping frames for each reference frame
    overlaps = []
    for ref_frame in ref_data:
        ref_frame_index, ref_start, ref_end = ref_frame
        
        ref_start += frame_offset
        ref_end += frame_offset
        
        
        # Do a first pass to find overlapping frames
        # i.e. frames that start before ref_end and end after ref_start
        overlap_data = [
            frame for frame in fp_data 
            if not (frame[3] <= ref_start or frame[2] >= ref_end)
        ]
        
        
        # Compute the fractional overlap 
        fractional_overlaps = []
        for frame in overlap_data:
            file, frame_index, frame_start, frame_end = frame
            
            overlap_start = max(ref_start, frame_start)
            overlap_end = min(ref_end, frame_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            frame_duration = frame_end - frame_start
            fraction_overlap = overlap_duration / frame_duration if frame_duration > 0 else 0
            row = [file, frame_index, fraction_overlap]
            fractional_overlaps.append(row)
            
        # If the overlap is negligible, skip it
        N_overlap_frames = np.sum([f[2] for f in fractional_overlaps])
        if N_overlap_frames <= overlap_threshold * N_frames_needed:
            overlaps.append( (ref_frame_index, []) )
        else:
            overlaps.append( (ref_frame_index, fractional_overlaps) )
    
    
    # Print the number of rejected frames per datacube
    overlap_percentages = []
    for i, (ref_frame_index, fractional_overlaps) in enumerate(overlaps):
        overlap_percent = np.sum([f[2] for f in fractional_overlaps]) / N_frames_needed
        overlap_percentages.append(overlap_percent)
    
    extrapolated_frames = np.sum( np.array(overlap_percentages) < overlap_threshold )
    if extrapolated_frames > 0:
        print(f'!!! WARNING !!! Datacube {reference_telemetry_file} has {extrapolated_frames} extrapolated frames (less than {overlap_threshold*100:0.0f}% overlap) that are rejected.')
    
    
    overlaps = Overlaps(dict(overlaps), len(ref_data), frame_offset)
    
    return overlaps



def temporally_sum_data(
    overlaps : 'Overlaps',
    data_dir : str, 
    dark_frame : np.ndarray | None = None
    ) -> np.ndarray:
    """
    Temporally sum data based on the provided overlaps.
    
    Parameters:
    -----------
    overlaps : Synchronizer.Overlaps
        A dict containing the overlapping frame information with structure
        {ref_frame_index : [overlapping_other_file, overlapping_frame_index, overlap_percentage]}
    data_dir : str
        Directory containing the data files.
    dark_file : ndarray | None
        dark frame to subtract from each frame.
        If None, no dark subtraction is performed.
        
    returns:
    --------
    summed_frames : ndarray
        A 3D array containing the temporally summed frames.
    """
    
    # This variable will hold the summed frames
    summed_frames = None
    
    for ref_index, overlapping_frames in overlaps.items():
        
        if len(overlapping_frames) == 0:
            continue
        
        if summed_frames is None:
            # Find the dimensions of the overlapping_data from the first frame
            dummy_frame_file = os.path.join(
                data_dir, 
                overlapping_frames[0][0].replace('.txt', '.fits')
            )
            # Use fitsio to memory map the file for efficiency
            dummy_frame = fitsio.FITS(dummy_frame_file, 'r')[0][0,:,:][0]
            # dummy_frame = fits.open(dummy_frame_file, memmap=True)[0].data[0]
            summed_frames = np.zeros((overlaps.ref_cubesize, *dummy_frame.shape))
        
        
        for overlap_frame, frame_index, overlap_percentage in overlapping_frames:
            filename = overlap_frame.replace('.txt', '.fits')
            filepath = os.path.join(data_dir, filename)
            # frame_data = fits.open(filepath, memmap=True)[0].data[frame_index]
            
            with fitsio.FITS(filepath, 'r') as fits_file:
                frame_data = fits_file[0][frame_index, :, :][0]
            
            if dark_frame is not None:
                frame_data = frame_data - dark_frame
                
            summed_frames[ref_index] += frame_data * overlap_percentage
        
        # print(f"  Processed reference frame {ref_index+1}/{overlaps.ref_cubesize}", end='\r')
    return summed_frames







class Overlaps(dict):
    """
    Class to hold overlap data with structure:
    
    {ref_frame_index : [overlapping_other_file, overlapping_frame_index, overlap_percentage]}
    
    Inherits from dict.
    
    Parameters:
    -----------
    input_dict : dict
        A dict containing the overlapping frame information with structure:
        {ref_frame_index : [overlapping_other_file, overlapping_frame_index, overlap_percentage]}
    ref_cubesize : int
        Number of frames in the reference datacube.
    frame_offset : float
        Time offset applied to the reference frames.
    *args, **kwargs : 
        Additional arguments passed to the dict constructor.
    
    
    Attributes:
    -----------
    ref_cubesize : int
        Number of frames in the reference datacube.
    frame_offset : float
        Time offset applied to the reference frames.
        
        
    Methods:
    --------
    to_dict() : dict
        Convert the overlaps object to a dictionary.
    save(output_file, overlaps) : None
        Save the overlap data to a json file.
    load(input_file) : dict
        Load the overlap data from a json file.
    """
    
    def __init__(
        self,
        input_dict={},
        ref_cubesize=np.nan, 
        frame_offset=0.,
        *args, **kwargs
        ):
        super().__init__(input_dict, *args, **kwargs)
        self.ref_cubesize = ref_cubesize
        self.frame_offset = frame_offset
        
    
    def to_dict(self):
        """
        Convert the overlaps object to a dictionary.
        
        returns:
        --------
        dict
            A dictionary representation of the overlaps object.
        """
        return {
            'meta': {
                'cubesize': self.ref_cubesize,
                'frame_offset': self.frame_offset
            },
            'data': dict(self)
        }
        
        
    def save(self, outfile):
        """
        Save the overlap data to a json file.
        
        Parameters:
        -----------
        output_file : str
            Path to the output json file.
        overlaps : dict
            A dict containing the overlapping frame information with structure
            {ref_frame_index : [overlapping_other_file, overlapping_frame_index, overlap_percentage]}
        """
        if not os.path.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))
            
        with open(outfile, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
            
            
            
    @classmethod
    def load(cls, infile):
        """
        Load the overlap data from a json file.
        
        Parameters:
        -----------
        infile : str
            Path to the input json file.
            
        returns:
        --------
        overlaps : dict
            A dict containing the overlapping frame information with structure
            {ref_frame_index : [overlapping_other_file, overlapping_frame_index, overlap_percentage]}
        """
        with open(infile, 'r') as f:
            obj = json.load(f)
            
        meta = obj.get('meta', {})
        data = obj.get('data', {})
        # Convert data keys back to int
        data = {int(key): val for key, val in data.items()}
        overlaps = cls(data)
        overlaps.ref_cubesize = meta.get('cubesize', np.nan)
        overlaps.frame_offset = meta.get('frame_offset', 0.)
        
        return overlaps
            
        
        
        