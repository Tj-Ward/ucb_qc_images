########
#   
#   I ported spm_smooth into python3!!!!!
#   
#   All credit goes to the SPM team. Please refer to them for mathematical explanation.
#  
#   WHY? scipy has a gaussian smooth implementation which works well but that implementation is a little different, it does not use 1st degree B-spline interpolation. When I discovered the difference, I could not find python implementations so I ported the matlab code. 
#  
#   Identical to SPM12 spm_smooth as of 10-21-2021 except the addition of edge-preservation option and you can return the nibabel image rather than having to save the output to a file.
#  
#   Method for edge preservation inspired by https://stackoverflow.com/a/36307291/7128154
#  
#   Enjoy!
#    -Tyler Ward
#  
#   import spm_smooth from this file
#   If you do not specify an output path, it will return a smoothed nibabel image
#   If you do specity an output path, it will save it to that path and not retun anything
#
#   Example:
#    img = loadnii(path)
#    img = spm_smooth(img,[8,8,8],edge_pres=False)
#
#
#   Changelog:
#      2023-06-02 - Added padding to edge of arrays. 8*smoothing kernel. 
#                   Prevents edge effects in tightly cropped images. 
#
#
#  
import nibabel as nib
import numpy as np
from scipy.special import erf
import os
from scipy.ndimage import convolve1d
def spm_smoothkern(fwhm,x,dtype,t=1): 
    eps =  np.finfo(dtype).eps
    s = (fwhm/np.sqrt(8*np.log(2)))**2+eps
    if t==0:
        raise Exception('Gaussian convolved with 0th degree B-spline NOT IMPLEMENTED')
    # Skipping this implementation as 1st degree b-spline is SPM default and I have no use for 0th degree 
    #  The below code has not finished porting but it's pretty easy to see how to do it. Idk that the 
    #  use case is for 0th.
    #if t==0
    #    % Gaussian convolved with 0th degree B-spline
    #    % int(exp(-((x+t))^2/(2*s))/sqrt(2*pi*s),t= -0.5..0.5)
    #    w1  = 1/sqrt(2*s);
    #    krn = 0.5*(erf(w1*(x+0.5))-erf(w1*(x-0.5)));
    #    krn(krn<0) = 0;
    if t==1:
        w1  =  0.5*np.sqrt(np.divide(2,s))
        w2  =  np.divide(-0.5,s)
        w3  =  np.sqrt(s/2/np.pi)
        krn =  0.5*(erf(w1*(x+1))*(x+1) + erf(w1*(x-1))*(x-1) - 2*erf(w1*x   )* x)+w3*(np.exp(w2*(x+1)**2)+ np.exp(w2*(x-1)**2)- 2*np.exp(w2*x**2))
        krn[krn<0] = 0
    else:
        raise Exception()
    return krn

def smooth1(P,s,output,edge_pres=False,dtype=np.float32):
    # VOX: array of xyz voxel size
    VOX = np.sqrt(np.sum((P.affine[:3, :3]) ** 2, axis=0),dtype=dtype)
    # Future work:
    # Add s to image description
    # I am having issue with str datatype. ADNI nifti's were created in matlab 
    #   and I think there may be incompatibilities when adding a string to descript
    #   ADNI images have no description so I could just overwrite it but I want this
    #   function to manipulate only what is necessary.
    # 
    # P.header['descrip'] 
    s  = np.divide(s,VOX,dtype=dtype)
    s1 = np.divide(s,np.sqrt(8*np.log(2)),dtype=dtype)

    x  = np.round(6*s1[0])
    x  = np.arange(-x,x+1,1)
    x  = spm_smoothkern(s[0],x,dtype,1);
    x  = x/np.sum(x)
    
    y  = np.round(6*s1[1])
    y  = np.arange(-y,y+1,1)
    y  = spm_smoothkern(s[1],y,dtype,1) 
    y  = y/np.sum(y)
    
    z  = np.round(6*s1[2])
    z  = np.arange(-z,z+1,1)
    z  = spm_smoothkern(s[2],z,dtype,1) 
    z  = z/np.sum(z)

    out = np.copy(P.get_fdata()).astype(dtype)
    pad_x,pad_y,pad_z = (np.array(s)*8).round(0).astype(int)
    out = np.pad(out, ((pad_x,pad_x),(pad_y,pad_y),(pad_z,pad_z)), mode='empty')

    nan_mask = np.isnan(out)
    out = np.nan_to_num(out)
    for i, k in enumerate((x,y,z)):
        out = convolve1d(out, k, axis=i)
    
    if edge_pres: 
        norm = np.ones(shape=out.shape)
        norm[nan_mask] = 0
        for i, k in enumerate((x,y,z)):
            norm = convolve1d(norm, k, axis=i)

        norm = np.where(norm==0, 1, norm)
        out = out/norm
    out[nan_mask] = np.nan
    out = out[pad_x:-pad_x,pad_y:-pad_y,pad_z:-pad_z]
    out_img = nib.Nifti1Image(out, P.affine.copy(), P.header.copy())
    out_img.header.set_data_dtype(dtype)
    return out_img

def spm_smooth(P,s,output=None,edge_pres=False,dtype=np.float64):
    # 3 dimensional convolution of an image
    if np.isin(type(s), [int,float]): s = [s, s, s]
    if type(P) == str:
        if os.path.exists(P):
            try:P = nib.load(P)
            except:raise ValueError('Input can not be read')
    elif type(P) != nib.Nifti1Image:
        raise ValueError('Input is not an image')    
    out_img = smooth1(P,s,output,edge_pres=edge_pres,dtype=dtype)
    if output == None:
        return out_img
    else:
        out_img.to_filename(output)
