# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 00:37:03 2020

@author: zhiqi
"""
import argparse

from frontalize_kalman import *


"""
This is the file containing the scripts for launching the frontlaization pipeline.
"""
def load_args():
    parser = argparse.ArgumentParser(description="Lip frontalization package")
    parser.add_argument('--profile_path', type=str, default="./Examples/trump.jpg", help="Path to the video/image to be frontalized")
    parser.add_argument('--save_dir', type=str, default=None, help="Directory to save the frontalized image")
    parser.add_argument('--save_filename', type=str, default=None, help="Filename for saving the frontalization image")
    parser.add_argument('--device', type=str, default='cuda', help="'cpu' or 'cuda'")
    parser.add_argument('--region', type=str, default='lip', help="'fz' for full size image, 'lip' for lip regions only, 'head' for head region")
    parser.add_argument('--frontal_path', type=str, default=None, help="Path the frontal ground truth image")
    parser.add_argument('--box_save_dir', type=str, default=None, help="The directory to save the bounding boxes")
    parser.add_argument('--gt_save_dir', type=str, default=None, help="The directory to save the corresponding ground truth region to compare with the reconstruction")
    parser.add_argument('--visible_only', type=int, default=0, help="0 for the raw tranformed image, 1 for the addtional visibility test"   )
    parser.add_argument('--zncc_path', type=str, default=None, help="File to store the zncc score of the input")
    parser.add_argument('--conv_visible', type=int, default=0, help="Whether to add conv visibility map")
    args = parser.parse_args()
   
    return args



def main():
    args = load_args()
    print(args)
    ROI = frontalize(profile_path = args.profile_path,
                        save_dir = args.save_dir,
                        save_filename = args.save_filename,
                        device = args.device, 
                        region = args.region,
                        frontal_path = args.frontal_path,
                        box_save_dir = args.box_save_dir,
                        gt_save_dir = args.gt_save_dir,
                        zncc_path = args.zncc_path,
                        visible_only = args.visible_only,
                        conv_visible = args.conv_visible)
                    

if __name__ == "__main__":
    main()
