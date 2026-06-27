"""
The goal of this code is to input psf telemetry files, PL telemetry files, 
and compute the overlapping PSF frames for every PL frame. 

Using this overlap, we will then create the time-synchronized PSF files. 


**Note: We assume telemetry files have the same naming scheme as their parent
    image files, eg. 'science_20240917_06:25:30.123456.fits' and
    'science_20240917_06:25:30.123456.txt' are the image and telemetry files
    respectively.


"""

import os
import sys
import argparse
import configparser
import numpy as np
from tqdm import tqdm
from astropy.io import fits


sys.path.append('./')
import data_handling
import Synchronizer
# import plotter


# Parse the command line arguments (config file path)
parser = argparse.ArgumentParser(
    description="Synchronize PSF data with the PL based on overlapping frames computed from telemetry files.")
parser.add_argument('config_file', type=str, 
                    help='Path to config file containing telemetry parameters')
args = parser.parse_args()

# Load config file
config = configparser.ConfigParser()
config.read(args.config_file)



OBJECT = config['Object']['name']
OBS_START = config['Object']['start_time']
OBS_END = config['Object']['end_time']
FP_TEL_DIR = config['FP']['tel_dir']
FP_IMAGE_DIR = config['FP']['image_dir']
FP_SYNC_DIR = config['FP']['synced_dir']
PL_TEL_DIR = config['PL']['tel_dir']
PL_IMAGE_DIR = config['PL']['image_dir']
# PL_FILE_TYPE = config['PL']['file_type']
PL_FILE_TYPE = '.fits'
FRAME_OFFSET = float(config['FP']['time_offset'])
OUTPUT_DIR = config['FP']['synced_dir']
PLOT_DIR = config['Plotting']['plot_dir']
OVERLAP_THRESH = float(config['FP']['overlap_threshold'])



if __name__ == '__main__':
    running_ascii = \
    """
============================================
SYNCHRONIZING PSF DATA WITH PHOTONIC LANTERN
============================================
"""
    print('\n\n\n'+running_ascii+'\n\n\n')
    
    # Find the relevant FP and PL image files. 
    # We will extract the data cube file names from these files. 
    # (we look in the fits files since the have they object name in the header)
    FP_image_files = data_handling.find_object_files(
        FP_IMAGE_DIR,
        OBJECT, 
        obs_start=OBS_START, obs_end=OBS_END
    )
    PL_image_files = data_handling.find_object_files(
        PL_IMAGE_DIR,
        OBJECT, 
        obs_start=OBS_START, obs_end=OBS_END, 
        # file_type=PL_FILE_TYPE
    )

    print(PL_image_files[0:10])
    print(FP_image_files[0:10])
    assert False
    # For testing, when the fits files do not exist but the telemetry files do:
    # You can run everything up to the data interpolation step.
    # FP_image_files = data_handling.find_object_files(
    #     FP_IMAGE_DIR,
    #     None, 
    #     obs_start=OBS_START, obs_end=OBS_END, 
    #     file_type='.txt'
    # )
    # PL_image_files = data_handling.find_object_files(
    #     PL_IMAGE_DIR,
    #     None, 
    #     obs_start=OBS_START, obs_end=OBS_END, 
    #     file_type='.txt'
    # )
    
    

    print(f"\n\nFound {len(FP_image_files)} fastcam image files and {len(PL_image_files)} slowcam image files for object {OBJECT}.\n\n")
    
    
    FP_telemetry_files = [f.replace('.fits', '.txt') for f in FP_image_files]
    PL_telemetry_files = [f.replace(PL_FILE_TYPE, '.txt') for f in PL_image_files]
    
    
    # Find the file times for every focal plane file
    FP_times = [data_handling.find_file_time(file) for file in FP_telemetry_files]
    PL_times = [data_handling.find_file_time(file) for file in PL_telemetry_files]
    
    
    
    
    # -------------------------------------------------------------------------
    # CREATE PSF SYNCHRONIZATION INSTRUCTION FILES
    # -------------------------------------------------------------------------
    for ref_file, overlapping_datacubes in Synchronizer.iter_overlapping_datacubes(
        PL_telemetry_files,
        FP_telemetry_files
    ):
        overlaps = Synchronizer.compute_fractional_overlaps(
            ref_file,
            overlapping_datacubes,
            ref_dir=PL_TEL_DIR,
            other_dir=FP_TEL_DIR,
            frame_offset=FRAME_OFFSET, 
            overlap_threshold=OVERLAP_THRESH
        )
        
        # Save this overlap data to file
        savefile = 'synced_fastcam_to_' + ref_file.replace('.txt', '.json')
        output_file = os.path.join(FP_SYNC_DIR, savefile)
        
        overlaps.save(output_file)
        
        
        
    
    # -------------------------------------------------------------------------
    # CREATE SYNCHRONIZED PSF DATA CUBES
    # -------------------------------------------------------------------------
    # For every PL data cube file, read in the corresponding synchronization
    # instruction file, and create the synchronized PSF data cube.
    
    print("\n\nCreating synchronized PSF data cubes...")
    for PL_image_file in tqdm(PL_image_files):
        datacube_file = PL_image_file.replace(PL_FILE_TYPE, '')
        sync_file = 'synced_fastcam_to_' + datacube_file + '.json'
        sync_filepath = os.path.join(FP_SYNC_DIR, sync_file)
        
        try: overlaps = Synchronizer.Overlaps.load(sync_filepath)
        except FileNotFoundError:
            print(f"No synchronization file found for {PL_image_file}, skipping.")
            continue
        
        synced_cube = Synchronizer.temporally_sum_data(
            overlaps,
            data_dir = FP_IMAGE_DIR
        )
        
        if synced_cube is None:
            print(f"No overlapping data found for {datacube_file}, skipping.")
            continue
        
        fits.writeto(
            os.path.join(FP_SYNC_DIR, 'synced_fastcam_to_' + datacube_file + '.fits'),
            synced_cube,
            overwrite=True
        )
    
    
    
    