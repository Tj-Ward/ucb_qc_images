import numpy as np
import nibabel as nib
import os, re, sys, glob
import time
import pathlib
import subprocess

from sklearn.linear_model import LinearRegression

import pandas as pd
import matplotlib.colors as Colors
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from scipy.stats import spearmanr

from nilearn.image import resample_to_img, resample_img, crop_img, smooth_img

# # This is a custom edge-preserving gaussian smoothing function
# # Open the code for details on how to use.
# import importlib.util
# spec = importlib.util.spec_from_file_location("spm_smooth", 'smoothing.py')
# smo = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(smo)


from PIL import Image, ImageOps, ImageFont, ImageDraw
from PIL.PngImagePlugin import PngInfo

from io import StringIO

import cv2

##############################################################################
#                        Definitions
#   Add tracers to the three sections below as more enter our studies
#

####################
#   Color Scale
# Add tracers here
####################
COLOR_SCALE = {
    "FTP": (0, 2.5),
    "MK6240": (0, 2.5),
    "PI2620": (0, 2.5),
    "GTP1": (0, 2.5),
    "PIB": (0, 2.7),
    "FBB": (0, 2.7),
    "FBP": (0, 2.7),
    "NAV": (0, 2.7),
    "AMYLOID": (0, 2.7),
}

####################
#   Reference ROI
# Add tracers here
####################

# Cerebellar GM ref region
CEREBGM = [
    "PIB",
]

# Whole Cerebellum ref region
WCEREB = ["FBB", "FBP", "NAV"]

# Inferior Cereb GM ref region
INFCEREB = ["PI2620", "MK6240", "FTP", "GTP1"]

####################
#   Primary ROI
# Add tracers here
####################
METAROI_TRACERS = ["PI2620", "MK6240", "FTP", "GTP1"]
CORTSUM_TRACERS = ["PIB", "FBB", "FBP", "NAV"]


########################
#    Useful Definitions
#########################

# Path to a font on the computer (this path is correct on Neurocluster):
# I specify a couple different font size to use in QC images.
# font_path = "/usr/share/fonts/gnu-free/FreeSans.ttf"
font_path = "Arial Unicode"
try:
    print(f"Using font: {font_path}")
    slice_number_font = ImageFont.truetype(font_path, size=24)
    scan_info_font = ImageFont.truetype(font_path, size=30)
    cbar_font = ImageFont.truetype(font_path, size=30)
except ValueError:
    font_paths = glob.glob("~/Library/Fonts/*.ttf")
    if len(font_paths) > 0:
        font_path = font_paths[0]
        print(f"Using font: {font_path}")
        slice_number_font = ImageFont.truetype(size=24)
        scan_info_font = ImageFont.truetype(size=30)
        cbar_font = ImageFont.truetype(size=30)

WM = [77, 2, 41]
BRAAK1 = [1006, 2006]
BRAAK2 = [17, 53]
BRAAK3 = [1016, 1007, 1013, 18, 2016, 2007, 2013, 54]
BRAAK4 = [
    1015,
    1002,
    1026,
    1023,
    1010,
    1035,
    1009,
    1033,
    2015,
    2002,
    2026,
    2023,
    2010,
    2035,
    2009,
    2033,
]
BRAAK5 = [
    1028,
    1012,
    1014,
    1032,
    1003,
    1027,
    1018,
    1019,
    1020,
    1011,
    1031,
    1008,
    1030,
    1029,
    1025,
    1001,
    1034,
    2028,
    2012,
    2014,
    2032,
    2003,
    2027,
    2018,
    2019,
    2020,
    2011,
    2031,
    2008,
    2030,
    2029,
    2025,
    2001,
    2034,
]
BRAAK6 = [1021, 1022, 1005, 1024, 1017, 2021, 2022, 2005, 2024, 2017]
META_TEMPORAL_CODES = [1006, 2006, 18, 54, 1007, 2007, 1009, 2009, 1015, 2015]

CORTICAL_SUMMARY_CODES = [
    1009,
    2009,
    1015,
    1030,
    2015,
    2030,
    1003,
    1012,
    1014,
    1018,
    1019,
    1020,
    1027,
    1028,
    1032,
    2003,
    2012,
    2014,
    2018,
    2019,
    2020,
    2027,
    2028,
    2032,
    1008,
    1025,
    1029,
    1031,
    2008,
    2025,
    2029,
    2031,
    1015,
    1030,
    2015,
    2030,
    1002,
    1010,
    1023,
    1026,
    2002,
    2010,
    2023,
    2026,
]


# This is just a list of cerebral GM ROIs plus amygdala which displays the outline of the brain
HEMISPHERE_OUTLINE_CODES_PATH = "required/QC_Image_Cortex_LUT.txt"
HEMISPHERE_OUTLINE_CODES = list(
    pd.read_csv(HEMISPHERE_OUTLINE_CODES_PATH)["Index"].astype(int)
)

WHOLE_CEREB_CODES = [46, 47, 8, 7]
CEREB_GM_CODES = [47, 8]

BRAINMASK_CODES = HEMISPHERE_OUTLINE_CODES + WHOLE_CEREB_CODES + WM

BRAIN_OUTLINE_CODES = HEMISPHERE_OUTLINE_CODES + CEREB_GM_CODES + [43, 4, 16]


########################
#    Custom Colormaps
#########################

# Colormap for PET images with alpha gradient so that it smoothly transitions to zero intensity
NIH_colors = np.genfromtxt("required/NIH.csv", delimiter=",", dtype=float)
NIH_colors[0:2, -1] = 0
NIH_colors[2:20, -1] = np.sqrt(np.linspace(0, 1, 18))
NIH = ListedColormap(NIH_colors)

# Colormap for colorbar (without alpha):
NIH_colors = np.genfromtxt("required/NIH.csv", delimiter=",", dtype=float)
NIH_noalpha = ListedColormap(NIH_colors)


def loadnii(path, orientation="LAS"):
    """
    Load nifti image with specified orientation
    """
    from nibabel.orientations import io_orientation, axcodes2ornt

    img = nib.load(path)
    img_ornt = io_orientation(img.affine)
    new_ornt = axcodes2ornt(orientation)
    img = img.as_reoriented(img_ornt)
    return img


def read_csv(filename, comment="#", sep=","):
    """
    Reads spreadsheets with comments in them

    Pandas read_csv will fail if there are # comments
    """
    lines = "".join([line for line in open(filename) if not line.startswith(comment)])
    return pd.read_csv(StringIO(lines), sep=sep)


def filter_gaussian(arr, affine, fwhm):
    """
    Gaussian filter, NOT edge preservation
    with bounding box
    """
    from scipy.ndimage import gaussian_filter

    affine = affine[:3, :3]  # Keep voxel scale to mm
    fwhm_over_sigma_ratio = np.sqrt(8 * np.log(2))
    vox_size = np.sqrt(np.sum(affine**2, axis=0))
    sigma = fwhm / (fwhm_over_sigma_ratio * vox_size)
    print("fwhm:", fwhm)
    print("sigma:", sigma)
    print("vox size:", vox_size)
    nan_mask = np.isnan(arr)
    gauss = np.copy(arr)
    # gauss[nan_mask] = 0
    gauss = gaussian_filter(
        gauss, sigma=sigma, order=0, mode="nearest", cval=0, truncate=10
    )
    # gauss[nan_mask] = 0

    return gauss


def create_metadata_dict(intensity_img_path, parcellation_path):
    """
    Loads the intensity image (probably a PET image) and a registered
     aparc+aseg from Freesurfer. Other parcellations work too.

    Includes:
     - The full image path (capturing analysis date in squid)
     - PET image modify date on Neurocluster
     - All aparc+aseg volumes
     - All aparc+aseg PET SUVRs

    """
    image = loadnii(intensity_img_path)
    image_array = image.get_fdata()

    parcellation = loadnii(parcellation_path)
    parcellation_array = np.round(parcellation.get_fdata(), 0)

    meta_dict = {}

    meta_dict["image"] = os.path.realpath(intensity_img_path)

    mod_time = time.ctime(os.path.getmtime(intensity_img_path))

    meta_dict["image_modify_date"] = mod_time

    if "aparc+aseg" in os.path.basename(intensity_img_path):
        mask_indencies = HEMISPHERE_OUTLINE_CODES
    else:
        mask_indencies = list(np.unique(parcellation_array))

    for mask_i in mask_indencies:
        if mask_i == 0:
            continue
        meta_intensity_key = f"{mask_i:.0f} PET"
        meta_size_key = f"{mask_i:.0f} VOL"

        meta_intensity = np.nanmean(image_array[parcellation_array == mask_i])
        meta_size = np.nansum(parcellation_array == mask_i)

        meta_dict[meta_intensity_key] = f"{meta_intensity:.3f}"
        meta_dict[meta_size_key] = f"{meta_size:.0f}"

    return meta_dict


def update_metadata_dict(pngimage, intensity_img_path, parcellation_path):
    """
    Accepts the path to a QC PNG image and will insert new metadata into it
     without recreating the QC image
    """
    print("Updating QC image metadata dictionary")
    metadata = PngInfo()
    image_meta = create_metadata_dict(intensity_img_path, parcellation_path)
    for dkey in list(image_meta.keys()):
        metadata.add_text(dkey, image_meta[dkey])

    print("Saving QC image:", pngimage)
    img = Image.open(pngimage)
    img.save(pngimage, pnginfo=metadata, optimize=True)


def check_img(qc_image, pet_img_path, aparc_path):
    """
    First validates image modify date has not changed. If it has'nt, return True. check_img passed.

    Second, create metadata and comare old to new using spearman correlation and difference checks.

    Return True if image is good.
    Return False if image must be re-created.
    """
    if not os.path.exists(qc_image):
        print("image_output does not exist:", qc_image)
        print("check_img returning False")
        return False

    try:
        existing_image_meta = Image.open(qc_image).text
    except Exception as e:
        print("Error reading metadata (it probably has no meta)")
        print("Can not check image.")
        return False

    filename_match = existing_image_meta["image"] == os.path.realpath(pet_img_path)

    mod_time = time.ctime(os.path.getmtime(pet_img_path))

    if existing_image_meta["image_modify_date"] == mod_time:
        print("Image modify date hasnt changed. No need to verify SUVR/VOL.")
        print("check_image returns True")
        return True

    expecting_image_meta = create_metadata_dict(pet_img_path, aparc_path)

    x_size = []
    y_size = []
    y_suvr = []
    x_suvr = []

    for meta_key in list(expecting_image_meta.keys()):
        if meta_key not in list(existing_image_meta.keys()):
            print("Warning")
            print("Missing key from metadata in QC image:", meta_key)
            print("Returning False, recreating QC image.")
            return False

        A = expecting_image_meta[meta_key]
        B = existing_image_meta[meta_key]

        if A.replace(".", "", 1).isdigit() and B.replace(".", "", 1).isdigit():
            A, B = float(A), float(B)
            if meta_key[-3:] == "VOL":
                x_size.append(A)
                y_size.append(B)
            if meta_key[-3:] == "PET":
                x_suvr.append(A)
                y_suvr.append(B)

    rsquare_size = spearmanr(np.nan_to_num(x_size), np.nan_to_num(y_size)).correlation
    rsquare_suvr = spearmanr(np.nan_to_num(x_suvr), np.nan_to_num(y_suvr)).correlation

    suvr_diff = np.nan_to_num(x_suvr) - np.nan_to_num(y_suvr)

    if np.nanmax(suvr_diff) > 0.1:
        print("SUVR difference =", np.nanmax(suvr_diff), "\nRerunning QC image.")
        return False

    x_suvr = np.asarray(x_suvr).reshape((-1, 1))
    y_suvr = np.asarray(y_suvr).reshape((-1, 1))

    lm = LinearRegression()
    lm.fit(x_suvr, y_suvr)
    Yhat = lm.predict(x_suvr)

    print("Number of values used in correlation:", len(x_size), len(y_suvr))
    print("SUVR spearman correlation:", rsquare_suvr)
    print("VOLUME spearman correlation:", rsquare_size)
    print("SUVR slope:", lm.coef_)
    print("SUVR y-intercept:", lm.intercept_)

    if (
        (rsquare_size >= 0.99)
        and (rsquare_suvr >= 0.99)
        and (np.round(lm.intercept_[0], 2) == 0)
        and (np.round(lm.coef_[0][0], 2) == 1)
    ):
        print("Correlation looks good... Not rerunning.")
        return True
    else:
        print("Correlation is not good enough. Rerunning.")
        return False


def findCropSlice(list_of_nibabel_images, pad=10, rtol=1e-08):
    """
    Use this to crop empty space out of images.

    Accepts a list of nibabel images or nifti paths and optimizes the crop so that all the data is included

    If you are running for a single image, just use nilearn.Image.crop_img

    Returns:
        slice_xyz: 3D slices determining the optimal slice to capture data in the images
        cropped_voxels: number of voxels in each dimension that gets sliced out
    """

    crop_list = []
    for image in list_of_nibabel_images:
        if type(image) == str:
            image = loadnii(image)
        image_tmp, crop_index = crop_img(image, pad=pad, return_offset=True, rtol=rtol)
        crop_list.append(crop_index)
        image_shape = image.shape

    slices_x = [i[0].indices(image_shape[0]) for i in crop_list]
    minx = min([i[0] for i in slices_x])
    minx = minx - pad if (minx - pad) > 0 else 0
    maxx = max([i[1] for i in slices_x])
    maxx = maxx + pad if (maxx + pad) < image_shape[0] else (image_shape[0] - 1)
    slicex = slice(minx, maxx, None)

    slices_y = [i[1].indices(image_shape[1]) for i in crop_list]
    miny = min([i[0] for i in slices_y])
    miny = miny - pad if (miny - pad) > 0 else 0
    maxy = max([i[1] for i in slices_y])
    maxy = maxy + pad if (maxy + pad) < image_shape[1] else (image_shape[1] - 1)
    slicey = slice(miny, maxy, None)

    slices_z = [i[2].indices(image_shape[2]) for i in crop_list]
    minz = min([i[0] for i in slices_z])
    minz = minz - pad if (minz - pad) > 0 else 0
    maxz = max([i[1] for i in slices_z])
    maxz = maxz + pad if (maxz + pad) < image_shape[2] else (image_shape[2] - 1)
    slicez = slice(minz, maxz, None)

    cropped_voxels = (minx, miny, minz)
    slice_xyz = (slicex, slicey, slicez)

    return slice_xyz, cropped_voxels


def loadImage(image, colormap, scale, alpha=1, resample=False):
    """
    path         : path to nii image
    colormap     : colormap to convert intensity into RGBA
    scale        : (MIN , MAX) example (0 , 2.5)
    pad          : number of voxels to pad sides
    resample     : False, nearest, linear

    Returns a 4D RGBA array with the correct scale and colormap

    """
    A = np.array(
        [
            [-1.0, 0.0, 0.0, 127.0],
            [0.0, 1.0, 0.0, -127.0],
            [0.0, 0.0, 1.0, -100.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    target_shape = [256, 256, 256]

    if type(image) == str:
        image = loadnii(image)

    image_shape = image.shape
    if resample:
        image = resample_img(
            image, target_affine=A, target_shape=target_shape, interpolation=resample
        )
    image_array = image.get_fdata()

    MIN, MAX = scale
    display_image = image_array.copy().astype(np.float32)
    display_image[display_image <= MIN] = MIN
    display_image[display_image >= MAX] = MAX
    display_image = np.round((display_image - MIN) / (MAX - MIN), 5)
    display_image = colormap(display_image, bytes=True)
    display_image[:, :, :, 3] = display_image[:, :, :, 3] * alpha

    return display_image


def loadAparc(image, alpha=1, indicies=False, resample=False):
    """
    image: aparc+aseg path or nibabel image
    indicies: Optionally, provide a list of FS indicies to select from the aparc+aseg
    resample: resample to 256^3 1mm vox size. Use when image is not isometric.

    Used to load and color a whole aparc+aseg atlas using Freesurfer LUT

    Similar to loadImage, it returns a 4D RGBA array.

    """
    A = np.array(
        [
            [-1.0, 0.0, 0.0, 127.0],
            [0.0, 1.0, 0.0, -127.0],
            [0.0, 0.0, 1.0, -100.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    target_shape = [256, 256, 256]
    color_LUT_dict = pd.read_csv(
        "required/FreeSurferColorLUT_brain.txt"
    )  # .set_index('Index')
    color_LUT_dict[["R", "G", "B", "A"]] = color_LUT_dict[["R", "G", "B", "A"]]  # /256
    color_LUT_dict["A"] = [
        255,
    ] * len(color_LUT_dict["A"])
    color_LUT_dict.loc[0, :] = [0, "Unknown", 0, 0, 0, 0]
    color_LUT_dict.loc[len(color_LUT_dict)] = [999999, "Error", 0, 0, 0, 0]
    color_LUT_dict = (
        color_LUT_dict[["Index", "R", "G", "B", "A"]]
        .set_index("Index")
        .astype(int)
        .T.to_dict("list")
    )

    if type(image) == str:
        image = loadnii(image)
    image_shape = image.shape
    if resample:
        image = resample_img(
            image, target_affine=A, target_shape=target_shape, interpolation="nearest"
        )
    image_array = np.asarray(image.get_fdata())
    image_array = np.round(image_array, 0)

    if indicies:
        indicies = list(np.array(indicies, ndmin=1))
        image_array[~np.isin(image_array, indicies)] = 0

    u, inv = np.unique(image_array, return_inverse=True)
    image_array = np.array([color_LUT_dict[x] for x in u])[inv].reshape(
        image_array.shape + (4,)
    )

    image_array[:, :, :, 3] = image_array[:, :, :, 3] * alpha
    image_array = image_array.astype(np.uint8)

    return image_array


def loadMask(image, rgba, indicies=1, resample=False):
    """
    path         : path to nii image or nibabel image
    rgba         : RGBA values as tuple
    indicies     : Indicies to use from mask, default is 1 for binary masks
    pad          : number of voxels to pad sides
    resample     : False, nearest, linear

    Returns a 4D RGBA array with the desired color.

    Outlining happens when creating 2D slices.

    """
    A = np.array(
        [
            [-1.0, 0.0, 0.0, 127.0],
            [0.0, 1.0, 0.0, -127.0],
            [0.0, 0.0, 1.0, -100.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    target_shape = [256, 256, 256]

    if type(image) == str:
        image = loadnii(image)

    image_shape = image.shape
    if resample:
        image = resample_img(
            image, target_affine=A, target_shape=target_shape, interpolation=resample
        )
    image_array = image.get_fdata()

    # Image is of type mask, assume integer values and round to 0.
    image_array = np.round(image_array, 0)

    display_image = image_array.copy()
    indicies = np.array(indicies, ndmin=1)
    display_image = np.isin(display_image, indicies).astype(np.uint8)

    # White cmap example:
    # ListedColormap(np.array([1,1,1,1]))
    display_image = np.repeat(display_image[:, :, :, np.newaxis], 4, axis=3)
    display_image[:, :, :, 3] = display_image[:, :, :, 3] * rgba[3]
    display_image[:, :, :, 2] = display_image[:, :, :, 2] * rgba[2]
    display_image[:, :, :, 1] = display_image[:, :, :, 1] * rgba[1]
    display_image[:, :, :, 0] = display_image[:, :, :, 0] * rgba[0]

    return display_image


def draw_image(image_2D, flip_LR=False):
    """
    Accepts RGBA 2D image
    Rotates so that image is visually correct orientation.
    Returns PIL image
    """
    if flip_LR:
        image_2D = np.flip(image_2D, 0)
    image_2D = np.rot90(image_2D, 1)
    image_2D = Image.fromarray(image_2D)
    return image_2D


def draw_mask(image_2D, outline_value=0, flip_LR=False):
    """
    Returns PIL image from RGBA array

    Can outline binary mask files by the number of voxels specified by "outline_value"
    """
    if flip_LR:
        image_2D = np.flip(image_2D, 0)
    image_2D = np.rot90(image_2D, 1)
    if outline_value != 0:
        image_2D[:, :, 3] = (image_2D[:, :, 3] > 0).astype(int) * 255
        vol = np.copy(image_2D)
        vol = np.sum(vol[:, :, :], axis=2)  # 2d mask
        vol = (vol > 0).astype(int)
        omask = np.zeros(vol.shape + (4,))

        cnts = cv2.findContours(
            vol.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(
                omask, [c], -1, (255, 255, 255, 255), thickness=outline_value
            )
        tmp_img = np.asarray(omask)
        image_2D[tmp_img == 0] = 0
    image_2D = Image.fromarray(image_2D.astype("uint8"), "RGBA")
    return image_2D


def addText(image_2D, text, location, font=slice_number_font):
    """
    Add text to a PIL image at a given location
    Used to insert slice numbers
    """
    if type(image_2D) != Image.Image:
        image_2D = Image.fromarray(image_2D.astype("uint8"), "RGBA")
    d = ImageDraw.Draw(image_2D)
    d.text(location, str(text), font=font, fill=(255, 255, 255, 255))
    return image_2D


def draw_mrifree(output, pet, tracer, display_path_name=False, image_date=False):
    """
     Primary function for MRI-free QC image creation

     Just proivide path to the output path to save png, path to pet img, and tracer type.

    This code is broken up into sections where each section adds a new piece to the image

    If you want to create QC images for a different project, copy this function and edit it
      to your visual needs.

    """

    template = "/home/jagust/adni/pipeline_scripts/Templates_ROIs/MRI-less_Pipeline/spmdefault_1mm_MNI_avg152T1.nii"
    NPDKA_ATLAS = "/home/jagust/adni/pipeline_scripts/Templates_ROIs/MRI-less_Pipeline/ADNI200_DK_Atlas/rADNICN200_aparc+aseg_smoo1.50_normalized.nii"
    NPDKA_INF_CEREB = "/home/jagust/adni/pipeline_scripts/Templates_ROIs/MRI-less_Pipeline/ADNI200_DK_Atlas/inf_cerebellar_npm.nii"
    GAAIN_CS = "/home/jagust/adni/pipeline_scripts/Templates_ROIs/MRI-less_Pipeline/Amyloid_ROIs/GAAIN_ROI/1mm_MRILESS_klunkvoi_ctx.nii"
    GAAIN_WC = "/home/jagust/adni/pipeline_scripts/Templates_ROIs/MRI-less_Pipeline/Amyloid_ROIs/GAAIN_ROI/1mm_MRILESS_klunkvoi_WhlCbl.nii"
    GAAIN_CG = "/home/jagust/adni/pipeline_scripts/Templates_ROIs/MRI-less_Pipeline/Amyloid_ROIs/GAAIN_ROI/1mm_MRILESS_klunkvoi_CerebGry.nii"

    if tracer in CEREBGM + WCEREB:
        MIN, MAX = 0.0, 2.7
    elif tracer in INFCEREB:
        MIN, MAX = 0.0, 2.5
    else:
        print("ERROR Error error")
        print(
            "Tracer (",
            tracer,
            ")",
            "does not have a reference region defined in this code!",
        )
        print("Add your tracer to the top of this code")
        return False
    if display_path_name == False:
        display_path_name = pet

    QC_IMAGE_WIDTH = 3000
    # Load all the images and masks

    nu_img = loadnii(template)
    pet_img = loadnii(pet)
    aparc_img = loadnii(NPDKA_ATLAS)
    GAAIN_CS_img = loadnii(GAAIN_CS)
    GAAIN_WC_img = loadnii(GAAIN_WC)
    GAAIN_CG_img = loadnii(GAAIN_CG)

    # Crop dimensions
    #           Sagittal[start...stop], Coronal[start...stop], Axial[start...stop]
    slice_xyz = (slice(5, 177, None), slice(5, 213, None), slice(5, 177, None))
    cropped_voxels = (
        slice_xyz[0].indices(pet_img.shape[0])[0],
        slice_xyz[1].indices(pet_img.shape[1])[0],
        slice_xyz[2].indices(pet_img.shape[2])[0],
    )

    nu_img = nu_img.slicer[slice_xyz[0], slice_xyz[1], slice_xyz[2]]
    pet_img = pet_img.slicer[slice_xyz[0], slice_xyz[1], slice_xyz[2]]
    aparc_img = aparc_img.slicer[slice_xyz[0], slice_xyz[1], slice_xyz[2]]
    GAAIN_CS_img = GAAIN_CS_img.slicer[slice_xyz[0], slice_xyz[1], slice_xyz[2]]
    GAAIN_WC_img = GAAIN_WC_img.slicer[slice_xyz[0], slice_xyz[1], slice_xyz[2]]
    GAAIN_CG_img = GAAIN_CG_img.slicer[slice_xyz[0], slice_xyz[1], slice_xyz[2]]

    if tracer in INFCEREB:
        infcereb_img = loadnii(NPDKA_INF_CEREB)
        infcereb_img = infcereb_img.slicer[slice_xyz[0], slice_xyz[1], slice_xyz[2]]

    ############################################
    #           Section 0
    #   Convert nibabel nii images into
    #       colored PIL arrays
    ############################################

    # Create MRI colored image
    nu_grey_img = loadImage(nu_img, cm.gray, (0.02, 1), alpha=1, resample=False)
    black_frame = nu_grey_img.copy()
    black_frame[:, :, :, :2] = 0

    # Create PET colored image
    pet_nih_alpha_img = loadImage(pet_img, NIH, (MIN, MAX), alpha=0.45, resample=False)
    pet_nih_noalpha_img = loadImage(
        pet_img, NIH, (MIN, MAX), alpha=0.95, resample=False
    )

    # Create aparc RGBA image for use in cortical ribon images

    QC_cortex_outline = loadMask(
        aparc_img, rgba=(255, 255, 255, 255), indicies=BRAIN_OUTLINE_CODES
    )

    # Create reference region colored images
    if tracer in INFCEREB:
        reference_region_img_filled = loadMask(infcereb_img, rgba=(255, 30, 30, 140))
        reference_region_img_outline = loadMask(infcereb_img, rgba=(255, 10, 10, 255))
    elif tracer in WCEREB:
        reference_region_img_filled = loadMask(GAAIN_WC_img, rgba=(255, 30, 30, 140))
        reference_region_img_outline = loadMask(GAAIN_WC_img, rgba=(255, 10, 10, 255))
    elif tracer in CEREBGM:
        reference_region_img_filled = loadMask(GAAIN_CG_img, rgba=(255, 30, 30, 140))
        reference_region_img_outline = loadMask(GAAIN_CG_img, rgba=(255, 10, 10, 255))
    else:
        print("ERROR Error error")
        print("Tracer (", tracer, ")", "is of unknown type!")
        return False

    # Create primary ROI colored images
    if tracer in METAROI_TRACERS:
        primary_roi_outline = loadMask(
            aparc_img, rgba=(255, 255, 255, 255), indicies=META_TEMPORAL_CODES
        )
    elif tracer in CORTSUM_TRACERS:
        primary_roi_outline = loadMask(GAAIN_CS_img, rgba=(255, 255, 255, 255))
    else:
        print("ERROR Error error")
        print("Tracer (", tracer, ")", "is of unknown type!")
        return False

    ############################################
    #              Section 1
    #   Define the image slice numbers
    ############################################
    # This can be expanded or reduced by just changing this value
    columns = 12
    image_dimensions = nu_img.shape

    applicable_slices = np.linspace(30, image_dimensions[2] - 41, columns, dtype=int)
    Axial_Slices = np.sort(applicable_slices)
    Display_Axial_Slices = Axial_Slices + cropped_voxels[2]

    applicable_slices = np.linspace(35, image_dimensions[1] - 36, columns, dtype=int)
    Coronal_Slices = np.sort(applicable_slices)
    Display_Coronal_Slices = Coronal_Slices + cropped_voxels[1]

    applicable_slices = np.linspace(35, image_dimensions[0] - 36, columns, dtype=int)
    Sagittal_Slices = np.sort(applicable_slices)
    Display_Sagittal_Slices = Sagittal_Slices + cropped_voxels[0]

    ############################################
    #              Section
    #      Draw axial MRI by itself
    ############################################

    axial_image_singles = []
    for slice_ in Axial_Slices:
        mri_frame = draw_image(nu_grey_img.take(slice_, axis=2))
        pet_frame = draw_image(pet_nih_alpha_img.take(slice_, axis=2))
        mri_frame.alpha_composite(pet_frame, (0, 0))
        axial_image_singles.append(mri_frame)

    axial_images = np.concatenate(axial_image_singles, axis=1)
    axial_images = Image.fromarray(axial_images.astype("uint8"), "RGBA")
    axial_images = ImageOps.scale(
        axial_images, (QC_IMAGE_WIDTH / axial_images.size[0]), resample=Image.LANCZOS
    )

    font_x_step = axial_images.size[0] // len(Display_Axial_Slices)
    for i, number in enumerate(Display_Axial_Slices):
        x_position = font_x_step * i
        axial_images = addText(
            axial_images, number, (x_position, 0), font=slice_number_font
        )

    border_between_groups = cm.gray(
        np.repeat(np.zeros(QC_IMAGE_WIDTH).reshape(1, -1), 50, 0), bytes=True
    )
    scan_summary = np.concatenate(
        [
            border_between_groups,
            axial_images,
        ],
        axis=0,
    )

    ############################################
    #              Section
    #      Draw PET with cortical ribbon
    ############################################

    def draw_bottom_rows(Slices, Display_Slices, pet, masks, axis):
        image_singles = []
        for slice_ in Slices:
            # frame = draw_image(nu_grey_img.take(slice_,axis=axis))
            frame = draw_image(black_frame.take(slice_, axis=axis))
            frame2 = draw_image(pet.take(slice_, axis=axis))
            frame.alpha_composite(frame2, (0, 0))
            for mask in masks:
                mask_frame = draw_mask(mask.take(slice_, axis=axis), outline_value=2)
                frame.alpha_composite(mask_frame, (0, 0))

            image_singles.append(frame)

        row = np.concatenate(image_singles, axis=1)

        row = Image.fromarray(row.astype("uint8"), "RGBA")
        row = ImageOps.scale(
            row, (QC_IMAGE_WIDTH / row.size[0]), resample=Image.LANCZOS
        )

        font_x_step = row.size[0] // len(Display_Slices)

        for i, number in enumerate(Display_Slices):
            x_position = font_x_step * i
            row = addText(row, number, (x_position, 0))
        return row

    r1 = draw_bottom_rows(
        Axial_Slices,
        Display_Axial_Slices,
        pet_nih_noalpha_img,
        [
            QC_cortex_outline,
        ],
        axis=2,
    )

    r2 = draw_bottom_rows(
        Coronal_Slices,
        Display_Coronal_Slices,
        pet_nih_noalpha_img,
        [
            QC_cortex_outline,
        ],
        axis=1,
    )
    r3 = draw_bottom_rows(
        Sagittal_Slices,
        Display_Sagittal_Slices,
        pet_nih_noalpha_img,
        [
            QC_cortex_outline,
        ],
        axis=0,
    )

    scan_summary = np.concatenate(
        [scan_summary, border_between_groups, r1, r2, r3], axis=0
    )

    # ############################################
    # #              Section
    # # Draw PET with primary ROI
    # #############################################
    r1 = draw_bottom_rows(
        Axial_Slices,
        Display_Axial_Slices,
        pet_nih_noalpha_img,
        [
            primary_roi_outline,
            reference_region_img_filled,
        ],
        axis=2,
    )
    r2 = draw_bottom_rows(
        Coronal_Slices,
        Display_Coronal_Slices,
        pet_nih_noalpha_img,
        [
            primary_roi_outline,
            reference_region_img_filled,
        ],
        axis=1,
    )
    r3 = draw_bottom_rows(
        Sagittal_Slices,
        Display_Sagittal_Slices,
        pet_nih_noalpha_img,
        [
            primary_roi_outline,
            reference_region_img_filled,
        ],
        axis=0,
    )

    scan_summary = np.concatenate(
        [scan_summary, border_between_groups, r1, r2, r3], axis=0
    )

    # ############################################
    # #               Section
    # #             Add colorbar
    # #############################################

    # Add colorbar to bottom
    cbar_size = scan_summary.shape[1] - 400

    cbar_height = 30
    cbar = NIH_noalpha(
        np.repeat(np.linspace(0, 1, cbar_size).reshape(1, -1), cbar_height, 0),
        bytes=True,
    )
    cbar[0:1, 0:, :] = [255, 255, 255, 255]
    cbar[-1:, 0:, :] = [255, 255, 255, 255]
    cbar[:, 0, :] = [255, 255, 255, 255]
    cbar[:, -1, :] = [255, 255, 255, 255]
    border = cm.gray(
        np.repeat(np.zeros(scan_summary.shape[1]).reshape(1, -1), 30, 0), bytes=True
    )
    bordertop = cm.gray(
        np.repeat(np.zeros(scan_summary.shape[1]).reshape(1, -1), 15, 0), bytes=True
    )
    sideborder = cm.gray(
        np.repeat(np.zeros(cbar_height).reshape(-1, 1), 200, 1), bytes=True
    )

    cbar = np.hstack((cbar, sideborder))
    cbar = np.hstack((sideborder, cbar))
    cbar = np.vstack((bordertop, cbar))
    cbar = np.vstack((cbar, border))
    # cbar = np.hstack((cbar,sideborder))
    # cbar = np.hstack((sideborder,cbar))
    cbar = Image.fromarray(cbar)
    d = ImageDraw.Draw(cbar)
    d.text((150, 15), str(MIN), font=cbar_font, fill=(255, 255, 255, 255))
    d.text(
        (cbar.size[0] - 180, 15),
        "≥" + str(MAX),
        font=cbar_font,
        fill=(255, 255, 255, 255),
    )
    cbar = np.asarray(cbar)

    scan_summary = np.vstack((scan_summary, cbar))

    # ############################################
    # #              Section
    # #      Add scan information text
    # #############################################

    # Add on summary info and sidebars
    border = cm.gray(
        np.repeat(np.zeros(20).reshape(1, -1), scan_summary.shape[0], 0), bytes=True
    )
    scan_summary = np.hstack((border, scan_summary, border))
    border = cm.gray(
        np.repeat(np.zeros(scan_summary.shape[1]).reshape(1, -1), 50, 0), bytes=True
    )

    border = Image.fromarray(border)
    d = ImageDraw.Draw(border)
    d.text(
        (20, 15), str(display_path_name), font=scan_info_font, fill=(255, 255, 255, 255)
    )
    if image_date:
        d.text(
            (border.size[0] - 340, 15),
            f"PET date: {image_date}",
            font=scan_info_font,
            fill=(255, 255, 255, 255),
        )
    border = np.asarray(border)

    scan_summary = np.vstack((border, scan_summary))

    scan_summary = Image.fromarray(scan_summary)
    scan_summary = scan_summary.convert("RGB")
    metadata = PngInfo()
    image_meta = create_metadata_dict(pet, NPDKA_ATLAS)
    for dkey in list(image_meta.keys()):
        metadata.add_text(dkey, image_meta[dkey])

    if output:
        print("Saving QC image:", output)
        scan_summary.save(output, pnginfo=metadata, optimize=True)
        return True
    else:
        return scan_summary


def draw_mridependent(
    output, images, tracer, xnat_paths=False, image_dates=False, include_metadata=True
):
    """
     Primary function for MRI-dependent QC image creation

    This code is broken up into sections where each section adds a new piece to the image

    If you want to create QC images for a different project, copy this function and edit it
      to your visual needs.

    xnat_paths - This list overwrites the paths to the list of images given.
    image_dates - ( MRI date, PET date)
    include_metadata - binary T/F. Including meta takes more time but is essential for deciding if the analysis has changed.
    """

    if len(images) == 3:
        nu, aparc, pet = images
    elif len(images) == 4:
        nu, aparc, pet, infcereb = images
    else:
        print("ERROR Error error")
        print('Wrong number of files for "images"')
        return False

    MIN, MAX = COLOR_SCALE[tracer]

    if xnat_paths == False:
        xnat_nu = nu
        xnat_aparc = aparc
        xnat_pet = pet
        if len(images) == 4:
            xnat_infcereb = infcereb
    else:
        xnat_nu = xnat_paths[0]
        xnat_aparc = xnat_paths[1]
        xnat_pet = xnat_paths[2]
        if len(images) == 4:
            xnat_infcereb = xnat_paths[3]

    QC_IMAGE_WIDTH = 3000
    # Load all the images and masks
    nu_img = nib.load(nu)
    pet_img = nib.load(pet)
    aparc_img = nib.load(aparc)

    # Take a brainmask using the aparc+aseg.
    # This is used to crop empty space out of the the 256^3 images
    brainmask_data = np.isin(aparc_img.get_fdata(), BRAINMASK_CODES).astype(float)
    # brainmask_data = filter_gaussian(brainmask_data,aparc_img.affine,[12,12,12])
    brainmask = nib.Nifti1Image(brainmask_data, aparc_img.affine, aparc_img.header)
    slice_xyz, cropped_voxels = findCropSlice(
        [
            brainmask,
        ],
        pad=14,
    )

    nu_img = nu_img.slicer[slice_xyz[0], slice_xyz[1], slice_xyz[2]]
    pet_img = pet_img.slicer[slice_xyz[0], slice_xyz[1], slice_xyz[2]]
    aparc_img = aparc_img.slicer[slice_xyz[0], slice_xyz[1], slice_xyz[2]]

    if tracer in INFCEREB:
        infcereb_img = nib.load(infcereb)
        infcereb_img = infcereb_img.slicer[slice_xyz[0], slice_xyz[1], slice_xyz[2]]

    ############################################
    #           Section 0
    #   Convert nibabel nii images into
    #       colored PIL arrays
    ############################################

    # Create MRI colored image
    nu_grey_img = loadImage(nu_img, cm.gray, (0, 200), alpha=1, resample=False)

    # Create PET colored image
    pet_nih_img = loadImage(pet_img, NIH, (MIN, MAX), alpha=0.6, resample=False)

    # Create whole brain colored images
    QC_cortex_outline = loadMask(
        aparc_img, rgba=(255, 255, 255, 255), indicies=HEMISPHERE_OUTLINE_CODES
    )

    # Create reference region colored images
    if tracer in INFCEREB:
        reference_region_img_filled = loadMask(
            infcereb_img, rgba=(255, 30, 30, 140), indicies=1
        )
        reference_region_img_outline = loadMask(
            infcereb_img, rgba=(255, 10, 10, 255), indicies=1
        )
    elif tracer in WCEREB:
        reference_region_img_filled = loadMask(
            aparc_img, rgba=(255, 30, 30, 140), indicies=WHOLE_CEREB_CODES
        )
        reference_region_img_outline = loadMask(
            aparc_img, rgba=(255, 10, 10, 255), indicies=WHOLE_CEREB_CODES
        )
    elif tracer in CEREBGM:
        reference_region_img_filled = loadMask(
            aparc_img, rgba=(255, 30, 30, 140), indicies=CEREB_GM_CODES
        )
        reference_region_img_outline = loadMask(
            aparc_img, rgba=(255, 10, 10, 255), indicies=CEREB_GM_CODES
        )
    else:
        print("ERROR Error error")
        print("Tracer (", tracer, ")", "is of unknown type!")
        return False

    # Create primary ROI colored images
    if tracer in METAROI_TRACERS:
        primary_roi_filled = loadMask(
            aparc_img, rgba=(255, 0, 0, 180), indicies=META_TEMPORAL_CODES
        )
        primary_roi_outline = loadMask(
            aparc_img, rgba=(255, 255, 255, 255), indicies=META_TEMPORAL_CODES
        )
    elif tracer in CORTSUM_TRACERS:
        primary_roi_filled = loadMask(
            aparc_img, rgba=(255, 0, 0, 180), indicies=CORTICAL_SUMMARY_CODES
        )
        primary_roi_outline = loadMask(
            aparc_img, rgba=(255, 255, 255, 255), indicies=CORTICAL_SUMMARY_CODES
        )
    else:
        print("ERROR Error error")
        print("Tracer (", tracer, ")", "is of unknown type!")
        return False

    ############################################
    #              Section 1
    #   Define the image slice numbers
    ############################################
    # This can be expanded or reduced by just changing this value
    columns = 12
    extra = 12
    # extra specifies how many extra slices we are taking outside of the columns
    # so that we can cut off the far ends, not displaying slices near the edge of plane.
    end = extra // 2

    image_dimensions = nu_img.shape

    for i in [0, 1, 2]:
        while columns + extra >= image_dimensions[i]:
            extra = extra - 2
            if extra <= 0:
                print("ERROR Error error:")
                print("Check segmentation of", xnat_paths)
                print(f"Brain is less than {columns} voxels in dimension {i}.")
                return False

    applicable_slices = np.linspace(0, image_dimensions[2], columns + extra, dtype=int)
    Axial_Slices = np.sort(applicable_slices)[end:-end]
    Display_Axial_Slices = Axial_Slices + cropped_voxels[2]

    applicable_slices = np.linspace(0, image_dimensions[1], columns + extra, dtype=int)
    Coronal_Slices = np.sort(applicable_slices)[end:-end]
    Display_Coronal_Slices = Coronal_Slices + cropped_voxels[1]

    applicable_slices = np.linspace(0, image_dimensions[0], columns + extra, dtype=int)
    Sagittal_Slices = np.sort(applicable_slices)[end:-end]
    Display_Sagittal_Slices = Sagittal_Slices + cropped_voxels[0]

    ############################################
    #              Section 2
    #      Draw axial MRI by itself
    ############################################

    axial_image_singles = []
    for slice_ in Axial_Slices:
        mri_frame = draw_image(nu_grey_img[:, :, slice_], flip_LR=True)
        axial_image_singles.append(mri_frame)

    axial_images = np.concatenate(axial_image_singles, axis=1)
    axial_images = Image.fromarray(axial_images.astype("uint8"), "RGBA")
    axial_images = ImageOps.scale(
        axial_images, (QC_IMAGE_WIDTH / axial_images.size[0]), resample=Image.LANCZOS
    )

    font_x_step = axial_images.size[0] // len(Display_Axial_Slices)
    for i, number in enumerate(Display_Axial_Slices):
        x_position = font_x_step * i
        axial_images = addText(axial_images, number, (x_position, 0))

    scan_summary = np.concatenate(
        [
            axial_images,
        ],
        axis=0,
    )

    ############################################
    #              Section 3
    #   Draw A/S/C MRI with masks ontop
    ############################################
    def draw_middle_rows(
        Slices, Display_Slices, mri, cortex_outline, reference_roi, axis
    ):
        image_singles = []
        for slice_ in Slices:
            mri_frame = draw_image(mri.take(slice_, axis=axis), flip_LR=True)

            cortex_outline_frame = draw_mask(
                cortex_outline.take(slice_, axis=axis), outline_value=1, flip_LR=True
            )
            mri_frame.alpha_composite(cortex_outline_frame, (0, 0))

            reference_roi_frame = draw_mask(
                reference_roi.take(slice_, axis=axis), outline_value=0, flip_LR=True
            )
            mri_frame.alpha_composite(reference_roi_frame, (0, 0))

            image_singles.append(mri_frame)
        row = np.concatenate(image_singles, axis=1)

        row = Image.fromarray(row.astype("uint8"), "RGBA")

        row = ImageOps.scale(
            row, (QC_IMAGE_WIDTH / row.size[0]), resample=Image.LANCZOS
        )

        font_x_step = row.size[0] // len(Display_Slices)
        for i, number in enumerate(Display_Slices):
            x_position = font_x_step * i
            row = addText(row, number, (x_position, 0))

        return row

    r1 = draw_middle_rows(
        Axial_Slices,
        Display_Axial_Slices,
        nu_grey_img,
        QC_cortex_outline,
        reference_region_img_filled,
        axis=2,
    )

    r2 = draw_middle_rows(
        Coronal_Slices,
        Display_Coronal_Slices,
        nu_grey_img,
        QC_cortex_outline,
        reference_region_img_filled,
        axis=1,
    )
    r3 = draw_middle_rows(
        Sagittal_Slices,
        Display_Sagittal_Slices,
        nu_grey_img,
        QC_cortex_outline,
        reference_region_img_filled,
        axis=0,
    )

    scan_summary = np.concatenate([scan_summary, r1, r2, r3], axis=0)

    # ############################################
    # #              Section 4
    # # Draw PET over MRI with eroded masks ontop
    # #############################################

    def draw_bottom_rows(
        Slices, Display_Slices, mri, pet, cortex_outline, reference_roi, axis
    ):
        image_singles = []
        for slice_ in Slices:
            mri_frame = draw_image(mri.take(slice_, axis=axis), flip_LR=True)

            pet_frame = draw_image(pet.take(slice_, axis=axis), flip_LR=True)
            mri_frame.alpha_composite(pet_frame, (0, 0))

            cortex_outline_frame = draw_mask(
                cortex_outline.take(slice_, axis=axis), outline_value=1, flip_LR=True
            )
            mri_frame.alpha_composite(cortex_outline_frame, (0, 0))

            reference_roi_frame = draw_mask(
                reference_roi.take(slice_, axis=axis), outline_value=2, flip_LR=True
            )
            mri_frame.alpha_composite(reference_roi_frame, (0, 0))

            image_singles.append(mri_frame)

        row = np.concatenate(image_singles, axis=1)

        row = Image.fromarray(row.astype("uint8"), "RGBA")
        row = ImageOps.scale(
            row, (QC_IMAGE_WIDTH / row.size[0]), resample=Image.LANCZOS
        )

        font_x_step = row.size[0] // len(Display_Slices)
        for i, number in enumerate(Display_Slices):
            x_position = font_x_step * i
            row = addText(row, number, (x_position, 0))

        return row

    r1 = draw_bottom_rows(
        Axial_Slices,
        Display_Axial_Slices,
        nu_grey_img,
        pet_nih_img,
        primary_roi_outline,
        reference_region_img_outline,
        axis=2,
    )

    r2 = draw_bottom_rows(
        Coronal_Slices,
        Display_Coronal_Slices,
        nu_grey_img,
        pet_nih_img,
        primary_roi_outline,
        reference_region_img_outline,
        axis=1,
    )
    r3 = draw_bottom_rows(
        Sagittal_Slices,
        Display_Sagittal_Slices,
        nu_grey_img,
        pet_nih_img,
        primary_roi_outline,
        reference_region_img_outline,
        axis=0,
    )

    scan_summary = np.concatenate([scan_summary, r1, r2, r3], axis=0)

    # ############################################
    # #              Section 6
    # #             Add colorbar
    # #############################################

    # Add colorbar to bottom
    cbar_size = scan_summary.shape[1] - 400

    cbar_height = 30
    cbar = NIH_noalpha(
        np.repeat(np.linspace(0, 1, cbar_size).reshape(1, -1), cbar_height, 0),
        bytes=True,
    )
    cbar[0:1, 0:, :] = [255, 255, 255, 255]
    cbar[-1:, 0:, :] = [255, 255, 255, 255]
    cbar[:, 0, :] = [255, 255, 255, 255]
    cbar[:, -1, :] = [255, 255, 255, 255]
    border = cm.gray(
        np.repeat(np.zeros(scan_summary.shape[1]).reshape(1, -1), 30, 0), bytes=True
    )
    bordertop = cm.gray(
        np.repeat(np.zeros(scan_summary.shape[1]).reshape(1, -1), 15, 0), bytes=True
    )
    sideborder = cm.gray(
        np.repeat(np.zeros(cbar_height).reshape(-1, 1), 200, 1), bytes=True
    )

    cbar = np.hstack((cbar, sideborder))
    cbar = np.hstack((sideborder, cbar))
    cbar = np.vstack((bordertop, cbar))
    cbar = np.vstack((cbar, border))
    # cbar = np.hstack((cbar,sideborder))
    # cbar = np.hstack((sideborder,cbar))
    cbar = Image.fromarray(cbar)
    d = ImageDraw.Draw(cbar)
    d.text((170, 15), str(MIN), font=cbar_font, fill=(255, 255, 255, 255))
    d.text(
        (cbar.size[0] - 180, 15),
        "≥" + str(MAX),
        font=cbar_font,
        fill=(255, 255, 255, 255),
    )
    cbar = np.asarray(cbar)

    scan_summary = np.vstack((scan_summary, cbar))

    # ############################################
    # #              Section 7
    # #      Add scan information text
    # #############################################

    # Add on summary info and sidebars
    border = cm.gray(
        np.repeat(np.zeros(20).reshape(1, -1), scan_summary.shape[0], 0), bytes=True
    )
    scan_summary = np.hstack((border, scan_summary, border))
    border = cm.gray(
        np.repeat(np.zeros(scan_summary.shape[1]).reshape(1, -1), 100, 0), bytes=True
    )

    border = Image.fromarray(border)
    d = ImageDraw.Draw(border)
    d.text((20, 15), str(xnat_nu), font=scan_info_font, fill=(255, 255, 255, 255))
    d.text((20, 55), str(xnat_pet), font=scan_info_font, fill=(255, 255, 255, 255))
    if image_dates:
        d.text(
            (border.size[0] - 340, 15),
            f"MRI date: {image_dates[0]}",
            font=scan_info_font,
            fill=(255, 255, 255, 255),
        )
        d.text(
            (border.size[0] - 340, 55),
            f"PET date: {image_dates[1]}",
            font=scan_info_font,
            fill=(255, 255, 255, 255),
        )
    border = np.asarray(border)

    scan_summary = np.vstack((border, scan_summary))

    try:
        scan_summary = Image.fromarray(scan_summary)
        scan_summary = scan_summary.convert("RGB")
        metadata = PngInfo()
        if include_metadata:
            image_meta = create_metadata_dict(xnat_pet, xnat_aparc)
            for dkey in list(image_meta.keys()):
                metadata.add_text(dkey, image_meta[dkey])

        print("Saving QC image:", output)
        scan_summary.save(output, pnginfo=metadata, optimize=True)

    except Exception as e:
        print(e)
        print("** ERROR: Could not save image", output)


def draw_fullAtlas(output, nu, aparc, pet, scale, xnat_paths=False, image_dates=False):
    """
    output: path to save png image
    nu/aparc/pet: nifti images in same space
    scale: tuple (min, max) display range for PET image
    xnat_paths: alternative filenames to feed QC image
    image_dates: Dates to put into QC images


    Primary function for MRI-dependent QC image creation

    This code is broken up into sections where each section adds a new piece to the image

    If you want to create QC images for a different project, copy this function and edit it
      to your visual needs.
    """
    MIN, MAX = scale

    if xnat_paths == False:
        xnat_nu = nu
        xnat_aparc = aparc
        xnat_pet = pet
    else:
        xnat_nu = xnat_paths[0]
        xnat_aparc = xnat_paths[1]
        xnat_pet = xnat_paths[2]

    QC_IMAGE_WIDTH = 3000
    # Load all the images and masks
    nu_img = loadnii(nu)
    pet_img = loadnii(pet)
    aparc_img = loadnii(aparc)

    brainmask_data = np.isin(aparc_img.get_fdata(), BRAINMASK_CODES).astype(float)
    # brainmask_data = filter_gaussian(brainmask_data,aparc_img.affine,[8,8,8])
    brainmask = nib.Nifti1Image(brainmask_data, aparc_img.affine, aparc_img.header)
    slice_xyz, cropped_voxels = findCropSlice(
        [
            brainmask,
        ],
        pad=5,
    )

    nu_img = nu_img.slicer[slice_xyz[0], slice_xyz[1], slice_xyz[2]]
    pet_img = pet_img.slicer[slice_xyz[0], slice_xyz[1], slice_xyz[2]]
    aparc_img = aparc_img.slicer[slice_xyz[0], slice_xyz[1], slice_xyz[2]]

    ############################################
    #           Section 0
    #   Convert nibabel nii images into
    #       colored PIL arrays
    ############################################

    # Create MRI colored image
    nu_grey_img = loadImage(nu_img, cm.gray, (0, 200), alpha=1, resample=False)

    # Create PET colored image
    pet_nih_img = loadImage(pet_img, NIH, (MIN, MAX), alpha=0.6, resample=False)

    fs_atlas = loadAparc(aparc_img, alpha=0.65)

    ############################################
    #              Section 1
    #   Define the image slice numbers
    ############################################
    # This can be expanded or reduced by just changing this value
    columns = 12
    extra = 10
    end = extra // 2

    image_dimensions = nu_img.shape

    for i in [0, 1, 2]:
        while columns + extra >= image_dimensions[i]:
            extra = extra - 2
            if extra <= 0:
                print("ERROR Error error:")
                print("Check segmentation of", xnat_paths)
                print(f"Brain is less than {columns} voxels in dimension {i}.")
                return False

    applicable_slices = np.linspace(
        0, image_dimensions[2] - 1, columns + extra, dtype=int
    )
    Axial_Slices = np.sort(applicable_slices)[end:-end]
    Display_Axial_Slices = Axial_Slices + cropped_voxels[2]

    applicable_slices = np.linspace(
        0, image_dimensions[1] - 1, columns + extra, dtype=int
    )
    Coronal_Slices = np.sort(applicable_slices)[end:-end]
    Display_Coronal_Slices = Coronal_Slices + cropped_voxels[1]

    applicable_slices = np.linspace(
        0, image_dimensions[0] - 1, columns + extra, dtype=int
    )
    Sagittal_Slices = np.sort(applicable_slices)[end:-end]
    Display_Sagittal_Slices = Sagittal_Slices + cropped_voxels[0]

    ############################################
    #               Section 2
    #       Draw axial MRI by itself
    ############################################
    """
    Anatomical variation results in rows of different width
    This is why we resize the row using ImageOps to a standard 3000px wide.
    With that, they can be stacked vertically without empty space.
    """

    axial_image_singles = []
    for slice_ in Axial_Slices:
        mri_frame = draw_image(nu_grey_img[:, :, slice_])
        axial_image_singles.append(mri_frame)

    axial_images = np.concatenate(axial_image_singles, axis=1)
    axial_images = Image.fromarray(axial_images.astype("uint8"), "RGBA")

    axial_images = ImageOps.scale(
        axial_images, (QC_IMAGE_WIDTH / axial_images.size[0]), resample=Image.LANCZOS
    )

    font_x_step = axial_images.size[0] // len(Display_Axial_Slices)
    for i, number in enumerate(Display_Axial_Slices):
        x_position = font_x_step * i
        axial_images = addText(axial_images, number, (x_position, 0))

    scan_summary = np.concatenate(
        [
            axial_images,
        ],
        axis=0,
    )

    ############################################
    #              Section 3
    #   Draw A/S/C MRI with masks ontop
    ############################################
    def draw_middle_rows(Slices, Display_Slices, mri, atlas, axis):
        image_singles = []
        for slice_ in Slices:
            mri_frame = draw_image(mri.take(slice_, axis=axis))

            roi_frame = draw_mask(atlas.take(slice_, axis=axis), outline_value=0)
            mri_frame.alpha_composite(roi_frame, (0, 0))

            image_singles.append(mri_frame)

        row = np.concatenate(image_singles, axis=1)
        row = Image.fromarray(row.astype("uint8"), "RGBA")

        row = ImageOps.scale(
            row, (QC_IMAGE_WIDTH / row.size[0]), resample=Image.LANCZOS
        )

        font_x_step = row.size[0] // len(Display_Slices)
        for i, number in enumerate(Display_Slices):
            x_position = font_x_step * i
            row = addText(row, number, (x_position, 0))

        return row

    r1 = draw_middle_rows(
        Axial_Slices, Display_Axial_Slices, nu_grey_img, fs_atlas, axis=2
    )

    r2 = draw_middle_rows(
        Coronal_Slices, Display_Coronal_Slices, nu_grey_img, fs_atlas, axis=1
    )
    r3 = draw_middle_rows(
        Sagittal_Slices, Display_Sagittal_Slices, nu_grey_img, fs_atlas, axis=0
    )

    scan_summary = np.concatenate([scan_summary, r1, r2, r3], axis=0)

    # ############################################
    # #              Section 4
    # # Draw PET over MRI with eroded masks ontop
    # #############################################

    def draw_bottom_rows(Slices, Display_Slices, mri, pet, axis):
        image_singles = []
        for slice_ in Slices:
            mri_frame = draw_image(mri.take(slice_, axis=axis))

            pet_frame = draw_image(pet.take(slice_, axis=axis))
            mri_frame.alpha_composite(pet_frame, (0, 0))

            image_singles.append(mri_frame)
        row = np.concatenate(image_singles, axis=1)

        row = Image.fromarray(row.astype("uint8"), "RGBA")

        row = ImageOps.scale(
            row, (QC_IMAGE_WIDTH / row.size[0]), resample=Image.LANCZOS
        )

        font_x_step = row.size[0] // len(Display_Slices)
        for i, number in enumerate(Display_Slices):
            x_position = font_x_step * i
            row = addText(row, number, (x_position, 0))

        return row

    r1 = draw_bottom_rows(
        Axial_Slices, Display_Axial_Slices, nu_grey_img, pet_nih_img, axis=2
    )

    r2 = draw_bottom_rows(
        Coronal_Slices, Display_Coronal_Slices, nu_grey_img, pet_nih_img, axis=1
    )
    r3 = draw_bottom_rows(
        Sagittal_Slices, Display_Sagittal_Slices, nu_grey_img, pet_nih_img, axis=0
    )

    scan_summary = np.concatenate([scan_summary, r1, r2, r3], axis=0)

    # ############################################
    # #              Section 6
    # #             Add colorbar
    # #############################################

    # Add colorbar to bottom
    cbar_size = scan_summary.shape[1] - 400

    cbar_height = 30
    cbar = NIH_noalpha(
        np.repeat(np.linspace(0, 1, cbar_size).reshape(1, -1), cbar_height, 0),
        bytes=True,
    )
    cbar[0:1, 0:, :] = [255, 255, 255, 255]
    cbar[-1:, 0:, :] = [255, 255, 255, 255]
    cbar[:, 0, :] = [255, 255, 255, 255]
    cbar[:, -1, :] = [255, 255, 255, 255]
    border = cm.gray(
        np.repeat(np.zeros(scan_summary.shape[1]).reshape(1, -1), 30, 0), bytes=True
    )
    bordertop = cm.gray(
        np.repeat(np.zeros(scan_summary.shape[1]).reshape(1, -1), 15, 0), bytes=True
    )
    sideborder = cm.gray(
        np.repeat(np.zeros(cbar_height).reshape(-1, 1), 200, 1), bytes=True
    )

    cbar = np.hstack((cbar, sideborder))
    cbar = np.hstack((sideborder, cbar))
    cbar = np.vstack((bordertop, cbar))
    cbar = np.vstack((cbar, border))
    # cbar = np.hstack((cbar,sideborder))
    # cbar = np.hstack((sideborder,cbar))
    cbar = Image.fromarray(cbar)
    d = ImageDraw.Draw(cbar)
    d.text((170, 15), str(MIN), font=cbar_font, fill=(255, 255, 255, 255))
    d.text(
        (cbar.size[0] - 180, 15),
        "≥" + str(MAX),
        font=cbar_font,
        fill=(255, 255, 255, 255),
    )
    cbar = np.asarray(cbar)

    scan_summary = np.vstack((scan_summary, cbar))

    # ############################################
    # #              Section 7
    # #      Add scan information text
    # #############################################

    # Add on summary info and sidebars
    border = cm.gray(
        np.repeat(np.zeros(20).reshape(1, -1), scan_summary.shape[0], 0), bytes=True
    )
    scan_summary = np.hstack((border, scan_summary, border))
    border = cm.gray(
        np.repeat(np.zeros(scan_summary.shape[1]).reshape(1, -1), 100, 0), bytes=True
    )

    border = Image.fromarray(border)
    d = ImageDraw.Draw(border)
    d.text((20, 15), str(xnat_nu), font=scan_info_font, fill=(255, 255, 255, 255))
    d.text((20, 55), str(xnat_pet), font=scan_info_font, fill=(255, 255, 255, 255))
    if image_dates:
        d.text(
            (border.size[0] - 340, 15),
            f"MRI date: {image_dates[0]}",
            font=scan_info_font,
            fill=(255, 255, 255, 255),
        )
        d.text(
            (border.size[0] - 340, 55),
            f"PET date: {image_dates[1]}",
            font=scan_info_font,
            fill=(255, 255, 255, 255),
        )
    border = np.asarray(border)

    scan_summary = np.vstack((border, scan_summary))

    try:
        scan_summary = Image.fromarray(scan_summary)
        scan_summary = scan_summary.convert("RGB")
        metadata = PngInfo()
        image_meta = create_metadata_dict(xnat_pet, xnat_aparc)
        for dkey in list(image_meta.keys()):
            metadata.add_text(dkey, image_meta[dkey])

        print("Saving QC image:", output)
        scan_summary.save(output, pnginfo=metadata, optimize=True)

    except Exception as e:
        print(e)
        print("** ERROR: Could not save image", output)


def draw_mridependent_TylerToueg(output, nu, aparc, names=False):
    font_path = "/usr/share/fonts/gnu-free/FreeSans.ttf"
    slice_number_font = ImageFont.truetype(font_path, 10)
    scan_info_font = ImageFont.truetype(font_path, 30)
    cbar_font = ImageFont.truetype(font_path, 30)

    WM = [77, 2, 41]
    BRAAK1 = [1006, 2006]
    BRAAK2 = [17, 53]
    BRAAK3 = [1016, 1007, 1013, 18, 2016, 2007, 2013, 54]
    BRAAK4 = [
        1015,
        1002,
        1026,
        1023,
        1010,
        1035,
        1009,
        1033,
        2015,
        2002,
        2026,
        2023,
        2010,
        2035,
        2009,
        2033,
    ]
    BRAAK5 = [
        1028,
        1012,
        1014,
        1032,
        1003,
        1027,
        1018,
        1019,
        1020,
        1011,
        1031,
        1008,
        1030,
        1029,
        1025,
        1001,
        1034,
        2028,
        2012,
        2014,
        2032,
        2003,
        2027,
        2018,
        2019,
        2020,
        2011,
        2031,
        2008,
        2030,
        2029,
        2025,
        2001,
        2034,
    ]
    BRAAK6 = [1021, 1022, 1005, 1024, 1017, 2021, 2022, 2005, 2024, 2017]
    META_TEMPORAL_CODES = [1006, 2006, 18, 54, 1007, 2007, 1009, 2009, 1015, 2015]

    CORTICAL_SUMMARY_CODES = [
        1009,
        2009,
        1015,
        1030,
        2015,
        2030,
        1003,
        1012,
        1014,
        1018,
        1019,
        1020,
        1027,
        1028,
        1032,
        2003,
        2012,
        2014,
        2018,
        2019,
        2020,
        2027,
        2028,
        2032,
        1008,
        1025,
        1029,
        1031,
        2008,
        2025,
        2029,
        2031,
        1015,
        1030,
        2015,
        2030,
        1002,
        1010,
        1023,
        1026,
        2002,
        2010,
        2023,
        2026,
    ]

    # This is just a list of cerebral GM ROIs plus amygdala which displays the outline of the brain
    HEMISPHERE_OUTLINE_CODES_PATH = "required/QC_Image_Cortex_LUT.txt"
    HEMISPHERE_OUTLINE_CODES = list(
        pd.read_csv(HEMISPHERE_OUTLINE_CODES_PATH)["Index"].astype(int)
    )

    WHOLE_CEREB_CODES = [46, 47, 8, 7]
    CEREB_GM_CODES = [47, 8]

    BRAINMASK_CODES = HEMISPHERE_OUTLINE_CODES + WHOLE_CEREB_CODES + WM

    BRAIN_OUTLINE_CODES = HEMISPHERE_OUTLINE_CODES + CEREB_GM_CODES + [43, 4, 16]
    FNT_POLE = [1032, 2032]
    PRS_OPER = [1018, 2018]
    PRS_ORBT = [1019, 2019]
    PERICAL = [1021, 2021]
    MDL_TMP = [1015, 2015]
    WHOLE = FNT_POLE + PRS_OPER + PRS_ORBT + PERICAL + MDL_TMP
    if names:
        xnat_paths = names
    else:
        xnat_paths = [nu, aparc]
    QC_IMAGE_WIDTH = 4000
    # Load all the images and masks
    nu_img = loadnii(nu)
    aparc_img = loadnii(aparc)

    # Take a brainmask using the aparc+aseg.
    # This is used to crop empty space out of the the 256^3 images
    brainmask_data = np.isin(aparc_img.get_fdata(), BRAINMASK_CODES).astype(float)
    # brainmask_data = filter_gaussian(brainmask_data,aparc_img.affine,[12,12,12])
    brainmask = nib.Nifti1Image(brainmask_data, aparc_img.affine, aparc_img.header)
    slice_xyz, cropped_voxels = findCropSlice(
        [
            brainmask,
        ],
        pad=10,
    )

    nu_img = nu_img.slicer[slice_xyz[0], slice_xyz[1], slice_xyz[2]]
    aparc_img = aparc_img.slicer[slice_xyz[0], slice_xyz[1], slice_xyz[2]]

    ############################################
    #           Section 0
    #   Convert nibabel nii images into
    #       colored PIL arrays
    ############################################

    # Create MRI colored image
    nu_grey_img = loadImage(nu_img, cm.gray, (0, 210), alpha=1, resample=False)

    # Create whole brain colored images
    ################################################################################################### colors
    QC_cortex_outline = loadMask(
        aparc_img, rgba=(255, 255, 255, 255), indicies=BRAIN_OUTLINE_CODES
    )

    # Create primary ROI colored images
    #
    # primary_roi_outline_1 = loadMask(aparc_img,rgba=(255,40,40,255),indicies=FNT_POLE)
    # primary_roi_outline_2 = loadMask(aparc_img,rgba=(250,41,255,255),indicies=PRS_OPER)
    # primary_roi_outline_3 = loadMask(aparc_img,rgba=(41,247,255,255),indicies=PRS_ORBT)
    # primary_roi_outline_4 = loadMask(aparc_img,rgba=(40,255,134,255),indicies=PERICAL)
    # primary_roi_outline_5 = loadMask(aparc_img,rgba=(255,200,40,255),indicies=MDL_TMP)

    # primary_roi_outline = primary_roi_outline_1+\
    #                         primary_roi_outline_2+\
    #                         primary_roi_outline_3+\
    #                         primary_roi_outline_4+\
    #                         primary_roi_outline_5

    primary_roi_outline = loadMask(aparc_img, rgba=(255, 20, 20, 255), indicies=WHOLE)

    ############################################
    #              Section 1
    #   Define the image slice numbers
    ############################################
    # This can be expanded or reduced by just changing this value
    columns = 20
    extra = 12
    # extra specifies how many extra slices we are taking outside of the columns
    # so that we can cut off the far ends, not displaying slices near the edge of plane.
    end = extra // 2

    image_dimensions = nu_img.shape

    for i in [0, 1, 2]:
        while columns + extra >= image_dimensions[i]:
            extra = extra - 2
            if extra <= 0:
                print("ERROR Error error:")
                print("Check segmentation of", xnat_paths)
                print(f"Brain is less than {columns} voxels in dimension {i}.")
                # return False

    applicable_slices = np.linspace(0, image_dimensions[2], columns + extra, dtype=int)
    Axial_Slices = np.sort(applicable_slices)[end:-end]
    Display_Axial_Slices = Axial_Slices + cropped_voxels[2]

    applicable_slices = np.linspace(0, image_dimensions[1], columns + extra, dtype=int)
    Coronal_Slices = np.sort(applicable_slices)[end:-end]
    Display_Coronal_Slices = Coronal_Slices + cropped_voxels[1]

    applicable_slices = np.linspace(0, image_dimensions[0], columns + extra, dtype=int)
    Sagittal_Slices = np.sort(applicable_slices)[end:-end]
    Display_Sagittal_Slices = Sagittal_Slices + cropped_voxels[0]

    ############################################
    #              Section 2
    #      Draw axial MRI by itself
    ############################################

    axial_image_singles = []
    for slice_ in Axial_Slices:
        mri_frame = draw_image(nu_grey_img[:, :, slice_], flip_LR=True)
        axial_image_singles.append(mri_frame)

    axial_images = np.concatenate(axial_image_singles, axis=1)
    axial_images = Image.fromarray(axial_images.astype("uint8"), "RGBA")
    axial_images = ImageOps.scale(
        axial_images, (QC_IMAGE_WIDTH / axial_images.size[0]), resample=Image.LANCZOS
    )

    font_x_step = axial_images.size[0] // len(Display_Axial_Slices)
    for i, number in enumerate(Display_Axial_Slices):
        x_position = font_x_step * i
        axial_images = addText(axial_images, number, (x_position, 0))

    scan_summary = np.concatenate(
        [
            axial_images,
        ],
        axis=0,
    )

    ############################################
    #              Section 3
    #   Draw A/S/C MRI with masks ontop
    ############################################
    def draw_middle_rows(Slices, Display_Slices, mri, cortex_outline, axis):
        image_singles = []
        for slice_ in Slices:
            mri_frame = draw_image(mri.take(slice_, axis=axis), flip_LR=True)

            cortex_outline_frame = draw_mask(
                cortex_outline.take(slice_, axis=axis), outline_value=1, flip_LR=True
            )
            mri_frame.alpha_composite(cortex_outline_frame, (0, 0))

            image_singles.append(mri_frame)
        row = np.concatenate(image_singles, axis=1)

        row = Image.fromarray(row.astype("uint8"), "RGBA")

        row = ImageOps.scale(
            row, (QC_IMAGE_WIDTH / row.size[0]), resample=Image.LANCZOS
        )

        font_x_step = row.size[0] // len(Display_Slices)
        for i, number in enumerate(Display_Slices):
            x_position = font_x_step * i
            row = addText(row, number, (x_position, 0))

        return row

    r1 = draw_middle_rows(
        Axial_Slices, Display_Axial_Slices, nu_grey_img, QC_cortex_outline, axis=2
    )

    r2 = draw_middle_rows(
        Coronal_Slices, Display_Coronal_Slices, nu_grey_img, QC_cortex_outline, axis=1
    )
    r3 = draw_middle_rows(
        Sagittal_Slices, Display_Sagittal_Slices, nu_grey_img, QC_cortex_outline, axis=0
    )

    scan_summary = np.concatenate([scan_summary, r1, r2, r3], axis=0)

    # ############################################
    # #              Section 4
    # # Draw PET over MRI with eroded masks ontop
    # #############################################

    def draw_bottom_rows(Slices, Display_Slices, mri, cortex_outline, axis):
        image_singles = []
        for slice_ in Slices:
            mri_frame = draw_image(mri.take(slice_, axis=axis), flip_LR=True)

            cortex_outline_frame = draw_mask(
                cortex_outline.take(slice_, axis=axis), outline_value=1, flip_LR=True
            )
            mri_frame.alpha_composite(cortex_outline_frame, (0, 0))

            image_singles.append(mri_frame)

        row = np.concatenate(image_singles, axis=1)

        row = Image.fromarray(row.astype("uint8"), "RGBA")
        row = ImageOps.scale(
            row, (QC_IMAGE_WIDTH / row.size[0]), resample=Image.LANCZOS
        )

        font_x_step = row.size[0] // len(Display_Slices)
        for i, number in enumerate(Display_Slices):
            x_position = font_x_step * i
            row = addText(row, number, (x_position, 0))

        return row

    r1 = draw_bottom_rows(
        Axial_Slices, Display_Axial_Slices, nu_grey_img, primary_roi_outline, axis=2
    )

    r2 = draw_bottom_rows(
        Coronal_Slices, Display_Coronal_Slices, nu_grey_img, primary_roi_outline, axis=1
    )
    r3 = draw_bottom_rows(
        Sagittal_Slices,
        Display_Sagittal_Slices,
        nu_grey_img,
        primary_roi_outline,
        axis=0,
    )

    scan_summary = np.concatenate([scan_summary, r1, r2, r3], axis=0)

    # Add on summary info and sidebars
    border = cm.gray(
        np.repeat(np.zeros(20).reshape(1, -1), scan_summary.shape[0], 0), bytes=True
    )
    scan_summary = np.hstack((border, scan_summary, border))
    border = cm.gray(
        np.repeat(np.zeros(scan_summary.shape[1]).reshape(1, -1), 100, 0), bytes=True
    )

    border = Image.fromarray(border)
    d = ImageDraw.Draw(border)
    d.text((20, 15), str(xnat_paths[0]), font=scan_info_font, fill=(255, 255, 255, 255))
    d.text((20, 55), str(xnat_paths[1]), font=scan_info_font, fill=(255, 255, 255, 255))
    border = np.asarray(border)

    scan_summary = np.vstack((border, scan_summary))
    border = cm.gray(
        np.repeat(np.zeros(scan_summary.shape[1]).reshape(1, -1), 20, 0), bytes=True
    )
    scan_summary = np.vstack((scan_summary, border))
    print("Saving QC image:", output)
    scan_summary = Image.fromarray(scan_summary)
    scan_summary.save(output, optimize=True)
