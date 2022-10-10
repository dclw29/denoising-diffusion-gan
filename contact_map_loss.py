"""
Rebuild a rudimentary structure from an image and calculate the loss between the contact map and the identified internal structure coordinates
Add to generator loss with less emphasis? Because the middle relative orientation is still important. Add directionality to it, i.e. can only minimise in one direction (bringing chains together)
Otherwise will it just push the chains further apart for the same goal?
How does this work with the intermediate noise as well?
"""

#TODO list:
# Add a damping effect, doesn't need to be exact.
# Decide where to include loss - weighting? - apply only at the end of an iteration cycle (i.e. when you get the pure version of the image out?)
# i.e. timestep == 4
# Investigate impact of noise during denoising
# Clip the loss to some maximum value if the image can't even return a valid map

import torch
import matplotlib.pyplot as plt
import numpy
import torch.nn.functional as F

class zmatrix_converter:
    """A coordinate converter class"""

    def __init__(self, zmatrix):
        self.zmatrix = zmatrix
        self.device = zmatrix.device

        # First atom 
        self.cartesian = torch.zeros((1, 3), device=device)

    def rotation_matrix(self, axis, angle):
        """
        Euler-Rodrigues formula for rotation matrix
        """
        # Normalize the axis
        axis = axis / torch.sqrt(torch.dot(axis, axis))
        a = torch.cos(angle / 2)
        b, c, d = -axis * torch.sin(angle / 2)
        # going to need to be careful with the maths here to make sure everything can be backpropagated
        A0 = a * a + b * b - c * c - d * d; A1 = 2 * (b * c - a * d); A2 = 2 * (b * d + a * c)
        B0 = 2 * (b * c + a * d); B1 = a * a + c * c - b * b - d * d; B2 = 2 * (c * d - a * b)
        C0 = 2 * (b * d - a * c); C1 = 2 * (c * d + a * b); C2 = a * a + d * d - b * b - c * c
        return torch.cat(( torch.cat( (A0.unsqueeze(0), A1.unsqueeze(0), A2.unsqueeze(0)) ).unsqueeze(0),
                           torch.cat( (B0.unsqueeze(0), B1.unsqueeze(0), B2.unsqueeze(0)) ).unsqueeze(0),
                           torch.cat( (C0.unsqueeze(0), C1.unsqueeze(0), C2.unsqueeze(0)) ).unsqueeze(0) ))

    def add_first_three_to_cartesian(self):
        """
        The first three atoms in the zmatrix need to be treated differently
        # remember first row of zmatrix is actually a tracker for N
        """

        # Second atom
        distance = zmatrix[1, 2].unsqueeze(0)
        self.cartesian = torch.cat( (self.cartesian, F.pad(input=distance.unsqueeze(0), pad=(0,2), mode='constant', value=0)) )

        # Third atom
        distance = zmatrix[2, 2].unsqueeze(0)
        angle = zmatrix[2, 4].unsqueeze(0)
        q = self.cartesian[1]  # position of atom 1
        r = self.cartesian[0]  # position of atom 2

        # Vector pointing from q to r
        a = r - q

        # Vector of length distance pointing along the x-axis
        d = distance * a / torch.sqrt(torch.dot(a, a))        

        # Rotate d by the angle around the z-axis
        d = self.rotation_matrix(torch.tensor([0., 0., 1.], device=device), angle.squeeze()) @ d.unsqueeze(-1)

        # Add d to the position of q to get the new coordinates of the atom
        p = q + d.squeeze() 
        self.cartesian = torch.cat( (self.cartesian, p.unsqueeze(0)) )

    def add_atom_to_cartesian(self, idx):
        """Find the cartesian coordinates of the atom"""
        bond_idx = self.zmatrix[idx, 1] -1 # 0 is reserved for "atomname"
        bond = self.zmatrix[idx, 2]
        angle_idx = self.zmatrix[idx, 3] -1
        angle = self.zmatrix[idx, 4]
        dihedral_idx = self.zmatrix[idx, 5] -1
        dihedral = self.zmatrix[idx, 6]

        q = self.cartesian[int(bond_idx.item())]
        r = self.cartesian[int(angle_idx.item())]
        s = self.cartesian[int(dihedral_idx.item())] 
   
        # Vector pointing from q to r
        a = r - q
        # Vector pointing from s to r
        b = r - s        

        # Vector of length distance pointing from q to r
        d = distance * a / torch.sqrt(torch.dot(a, a))
        # Vector normal to plane defined by q, r, s
        normal = torch.cross(a, b)

        # Rotate d by the angle around the normal to the plane defined by q, r, s
        d = self.rotation_matrix(normal, angle) @ d.unsqueeze(-1)
        
        # Rotate d around a by the dihedral
        d = self.rotation_matrix(a, dihedral) @ d

        # Add d to the position of q to get the new coordinates of the atom
        p = q + d.squeeze() 
        self.cartesian = torch.cat( (self.cartesian, p.unsqueeze(0)) )

    def zmatrix_to_cartesian(self):
        """
        Convert the zmartix to cartesian coordinates
        """
        self.add_first_three_to_cartesian()

        for atom in range(3, len(self.zmatrix)):
            self.add_atom_to_cartesian(atom)

        return self.cartesian

def plot_sample(S):
    """
    Based on a sample produced by the generator, plot it
    """
    #S = sample[0] # get the first sample in the batch
    S = S.permute(1, 2, 0)
    plt.imshow(S.detach().cpu().numpy())
    plt.show()

def shift_idx(idx_arr, i, floor):
    """
    Shift the individual idx values based on its position to get a true zmatrix
    floor represents the border regions of null values 
    Taken from data2pdb
    """
    if idx_arr.round() == floor:
        return i + idx_arr.round() + 2 # set to minus 1 normally (move to zero)
    else:
        return i + idx_arr.round() + 1 # weird extra +1 shift needed

def res_256_to_64(image):
    """
    Convert an image with 256 resolution to 64 by averaging all pixel values in 8 x 8 squares
    """
    image_reduced = torch.nn.functional.avg_pool2d(image, kernel_size=(4,4)) # not median mind you
    return image_reduced

def remove_borders(arr):
    """
    Remove the zero border region from an image
    1 at the top, 1 at the bottom and 6 at the sides
    Also merge channels, input is BxCxHxW
    """
    # slice
    arr = arr[:, :, :, 6:-6]
    arr = arr[:, :, 1:-1, :]

    arr_dist = arr[:, 3] 
    arr_struc = arr[:, :3]

    # arr has third axis we need to add together intensities (for colour images), no 255 because these aren't images
    return torch.sum(arr_struc, dim=1) / 3, arr_dist[:, 5:-5] # cut top and bottom of distance map also

def build_sidechain_shift_array(shift, shape):
    """
    Build a shift array that can be added to the backbone information to include information on current sidehcians in the relative atom indexing
    Use based on current atom shift value, and the shape needed of the array (likely 6 x 4)
    """
    zero = torch.zeros_like(shape)
    zero[:, 0::2] += shift
    return zero

def cut_sidechain_pixels(sidechain_pixels):
    """
    Cut data based on border region begins (-1)
    """
    diff = torch.sum(sidechain_pixels.detach(), dim=1)
    D_slice = diff < 0.1 # assuming the sum is good enough to discount erroneous molecules (and the model isn't introducing fake atoms)
    return sidechain_pixels[~D_slice]

def build_sidechains(backbone, sidechain_raw, diag, Dmin, idx_diag, idx_min, dimer=False, dimer_shift=0):
    """
    Build the sidechain data from all the remainder of the input image. These then need to be inserted into the backbond data
    (with appropiate shifting of numbers), and unnormalised
    dimer_shift is the shift needed if this is the second chain (chain B) in the structure for the indexing
    Floor (in code) keeps track of the shift needed for the sidechain atoms (from the backbone and sidechain atoms being added in)
    """

    device = backbone.device

    # we're assuming that sidechain length (non black regions) is correct for the amino acid
    floor = 0

    # unnormalise floor values for idx shifting
    if dimer:
        floor += idx_min-1 + dimer_shift # plus dimer_shift for second chain - dependent on the number of atoms in the first chain
    else:
        floor += idx_min-1

    # get every three up to the length of the sidechain from sidechain_raw
    # we need to shift by the current backbone bonded to etc. coordinate
    new_chain = torch.empty((1, 7), device=device)
    sidechain_shift = 0 # how many atoms to shift forward by based on atoms being added from sidechains
    #for cnt, A in enumerate(AA):
    for cnt in range(13): # always loop through 13 times (number of unique resid)
        new_addition = backbone[cnt*4 : cnt*4 + 4] + build_sidechain_shift_array(sidechain_shift, backbone[cnt*4 : cnt*4 + 4]).to(device)
        atomnames = torch.zeros((len(new_addition), 1)).to(device)
        atomnames[0] = 1
        new_addition = torch.cat((atomnames, new_addition), dim=1)
        new_chain = torch.cat((new_chain, new_addition), dim=0)
        floor += 4
  
        if len(cut_sidechain_pixels(sidechain_raw[: , cnt*4 : cnt*4 + 3])) != 0: # catch the GLY

            # read left to right, top to bottom
            if dimer:
                sidechain_pixels = cut_sidechain_pixels(sidechain_raw[: , cnt*4 : cnt*4 + 3])
                bond = sidechain_pixels[1::2, 0] # remember this needs to be flipped in chain A
                bond_idx = sidechain_pixels[::2, 0]
                angle = sidechain_pixels[1::2, 1] 
                angle_idx = sidechain_pixels[::2, 1]
                dihedral = sidechain_pixels[1::2, 2] 
                dihedral_idx = sidechain_pixels[::2, 2]
            else:
                sidechain_pixels = cut_sidechain_pixels(sidechain_raw[: , cnt*4 : cnt*4 + 3])
                bond = sidechain_pixels[::2, 0].flip(dims=(0,))
                bond_idx = sidechain_pixels[1::2, 0].flip(dims=(0,))
                angle = sidechain_pixels[::2, 1].flip(dims=(0,)) 
                angle_idx = sidechain_pixels[1::2, 1].flip(dims=(0,))
                dihedral = sidechain_pixels[::2, 2].flip(dims=(0,))
                dihedral_idx = sidechain_pixels[1::2, 2].flip(dims=(0,))
    
            bond = ((bond)  *  diag[0]) + Dmin[0]
            angle = ((angle)  *  diag[1]) + Dmin[1]
            dihedral = ((dihedral)  *  diag[2]) + Dmin[2]

            if dimer:
                # idx values
                bond_idx =  (bond_idx * idx_diag) + floor
                angle_idx = (angle_idx * idx_diag) + floor
                dihedral_idx = (dihedral_idx * idx_diag) + floor
            else:
                # idx values
                bond_idx =  (bond_idx * idx_diag) + floor
                angle_idx = (angle_idx * idx_diag) + floor
                dihedral_idx = (dihedral_idx * idx_diag) + floor

            # shift idx values to give true zmatrix
            bond_idx_shift = torch.empty(1, device=device); angle_idx_shift = torch.empty(1, device=device); dihedral_idx_shift = torch.empty(1, device=device)
            for i in range(len(bond_idx)):
                bond_idx_shift = torch.cat((bond_idx_shift, shift_idx(bond_idx[i], i, floor).unsqueeze(0)))
                angle_idx_shift = torch.cat((angle_idx_shift, shift_idx(angle_idx[i], i, floor).unsqueeze(0)))
                dihedral_idx_shift = torch.cat((dihedral_idx_shift, shift_idx(dihedral_idx[i], i, floor).unsqueeze(0)))
            bond_idx_shift = bond_idx_shift[1:]; angle_idx_shift = angle_idx_shift[1:]; dihedral_idx_shift = dihedral_idx_shift[1:]

            sidechains = torch.cat ( (bond_idx_shift.unsqueeze(-1), bond.unsqueeze(-1), angle_idx_shift.unsqueeze(-1), angle.unsqueeze(-1), 
                                      dihedral_idx_shift.unsqueeze(-1), dihedral.unsqueeze(-1)), dim=1
            )

            atomnames_t = torch.zeros((len(sidechains), 1)).to(device)
            sidechains = torch.cat ( (atomnames_t, sidechains), dim=1)

            sidechain_shift += len(sidechains)
            floor += len(sidechains)
            new_chain = torch.cat( (new_chain, sidechains) , dim=0)
    
    return new_chain[1:]

def convert_cwd_backbone(bond_idx, bond, angle_idx, angle, dihedral_idx, dihedral, diag, Dmin, idx_diag, idx_min, dimer=False, dimer_shift=0):
    """
    Convert the standardised coordinates back into real values that are useful for our structures
    Taken from data2pdb
    Specifically get the backbone coordinates, onto which we can build the sidechains
    :param bond_idx:  coordinates normalised output 
    :param bond:  coordinates normalised output 
    :param angle_idx:  coordinates normalised output 
    :param angle:  coordinates normalised output 
    :param dihedral_idx:  coordinates normalised output 
    :param dihedral:  coordinates normalised output     
    :param dimer shift: how much to shift idx values by - note this depends on the first chain that is generated
    :returns: zmatrix lines
    """

    device = bond_idx.device

    # internal coord values
    bond = ((bond)  *  diag[0]) + Dmin[0]
    angle = ((angle)  *  diag[1]) + Dmin[1]
    dihedral = ((dihedral)  *  diag[2]) + Dmin[2]

    if dimer:
        # idx values
        floor = idx_min-1 + dimer_shift # plus dimer_shift for second chain - dependent on the number of atoms in the first chain
        bond_idx =  (bond_idx * idx_diag) + floor
        angle_idx = (angle_idx * idx_diag) + floor
        dihedral_idx = (dihedral_idx * idx_diag) + floor
    else:
        # idx values
        floor = idx_min-1
        bond_idx =  (bond_idx * idx_diag) + floor
        angle_idx = (angle_idx * idx_diag) + floor
        dihedral_idx = (dihedral_idx * idx_diag) + floor

    # shift idx values to give true zmatrix
    bond_idx_shift = torch.empty(1, device=device); angle_idx_shift = torch.empty(1, device=device); dihedral_idx_shift = torch.empty(1, device=device)
    for i in range(len(bond_idx)):
        bond_idx_shift = torch.cat((bond_idx_shift, shift_idx(bond_idx[i], i, floor).unsqueeze(0)))
        angle_idx_shift = torch.cat((angle_idx_shift, shift_idx(angle_idx[i], i, floor).unsqueeze(0)))
        dihedral_idx_shift = torch.cat((dihedral_idx_shift, shift_idx(dihedral_idx[i], i, floor).unsqueeze(0)))
    bond_idx_shift = bond_idx_shift[1:]; angle_idx_shift = angle_idx_shift[1:]; dihedral_idx_shift = dihedral_idx_shift[1:]

    cwd = torch.cat((torch.cat((torch.cat((torch.cat((torch.cat((bond_idx_shift.unsqueeze(-1), bond.unsqueeze(-1)), dim=-1),
                     angle_idx_shift.unsqueeze(-1)), dim=-1),
                     angle.unsqueeze(-1)), dim=-1), dihedral_idx_shift.unsqueeze(-1)), dim=-1),
                     dihedral.unsqueeze(-1)), dim=-1)

    return cwd

def convert_zmat(X, params):
    """
    The convert zmat function from your general approach, but written in pytorch
    To calculate true deviations from the contact map, we will need to unnormalise, hence params (as torch tensor though)
    """

    diagA, DminA, idxminA, diagB, DminB, idxminB, diag_middle, Dmin_middle, idx_diag = params

    # get centre of mass of side chains just from points! (we know rough cardinal ordering with where CA needs to go, so ignore every other atom when parsing later)
    # We can't really penalise when it doesn't get the amino acid symbol correct (without a MSE error loss or some classification)

    # we do need relative orientation data
    for x in X:

        # normalise like the image would be (i.e. shift to be between 0 and 1)
        x = (x + 1) / 2 # standard normalisation regardless of the actual values of x (shouldn't be greater than 1 by definition!)

        #######################################################################################################
        ##### GET RELATIVE ORIENTATION OF HELICES FROM MIDDLE ROW #####
        # Unnormalise the middle row - we actually write the coordinates through in the zmatrix writing stage
        middle = x[30:32, 2:-2]
        # average across the y dimension
        middle = torch.mean(middle, axis=0)
        # this "spread" needs to be treated more carefully - maybe just the nonzero size because will it vary?
        #middle = middle[middle != 0]
        spread = int(48/6)  # (already removed the initial padding in gen_moleucule_dimer) - should be fixed 136 in initial attempts
        middle_bond = middle[0:spread].mean()
        middle_angleA = middle[spread : spread*2].mean(); middle_angleB = middle[spread*2 : spread*3].mean()
        middle_dihedralA = middle[spread*3 : spread*4].mean(); middle_dihedralB = middle[spread*4 : spread*5].mean(); middle_dihedralC = middle[spread*5: spread*6].mean()
        # now rejoin angle and dihedral
        middle_angle = torch.cat((middle_angleA.unsqueeze(0), middle_angleB.unsqueeze(0)))
        middle_dihedral = torch.cat((middle_dihedralA.unsqueeze(0), middle_dihedralB.unsqueeze(0), middle_dihedralC.unsqueeze(0)))
        # all variables need to be averaged individually before being renormalised
        middle_bond = (middle_bond * diag_middle[0]) + Dmin_middle[0]
        middle_angle = (middle_angle * diag_middle[1]) + Dmin_middle[1]
        middle_dihedral = (middle_dihedral * diag_middle[2]) + Dmin_middle[2]

        ########################################################################################################
        ##### GET REMAINING COORDINATE DATA #####
    
        ####### Continue with remaining unnormalisation
        # Get chain A first
        backbone_cwdA = convert_cwd_backbone(x[20], x[21], x[22], x[23], x[24], x[25], diagA, DminA, idx_diag, idxminA, dimer=False)
        cwdA = build_sidechains(backbone_cwdA, x[:20], diagA, DminA, idx_diag, idxminA)
        # don't forget we have mirror on bottom, so move in reverse to pickup pixel values
        backbone_cwdB = convert_cwd_backbone(x[41], x[40], x[39], x[38], x[37], x[36], diagB, DminB, idx_diag, idxminB, dimer=True, dimer_shift = len(cwdA)) 
        cwdB = build_sidechains(backbone_cwdB, x[42:], diagB, DminB, idx_diag, idxminB, dimer=True, dimer_shift = len(cwdA))
        # if the residue-structure relationship is broken, this may return bad results

        # based on where the last backbone atoms are in atoms_A, we can establish how to build chain B of the TM domain
        # this depends a lot on the networks ability to generate the correct residue coordinate relationships
        N_pos = torch.where(cwdA[:, 0] == 1)[0][-1].unsqueeze(0)
        CA_pos = N_pos + 1; C_pos = N_pos + 2
        # plus 1 to move from pythonic indexing
        middle = torch.cat(( torch.tensor(1., device=device).unsqueeze(0), C_pos + 1, middle_bond.unsqueeze(0), CA_pos + 1, middle_angle[0].unsqueeze(0), N_pos + 1, middle_dihedral[0].unsqueeze(0),
                             C_pos + 1, middle_angle[1].unsqueeze(0), CA_pos + 1, middle_dihedral[1].unsqueeze(0),
                             C_pos + 1, middle_dihedral[2].unsqueeze(0) ))

        # now convert into coordinate system with zmatrix method
        zmatrix = torch.zeros(1, 7).to(device) # first atom can be zeros
        # we will enforce our conditions for the starting coordiantes
        for cnt, c in enumerate(cwdA[1:], start=1):
            if cnt==1:
                zmatrix = torch.cat( (zmatrix, torch.cat( (c[:3].unsqueeze(0), torch.zeros(1, 4).to(device)), dim=1)) , dim=0)
                #f.write("N  %i  %f\n"%(int(c[0]), c[1]))
            elif cnt==2:
                zmatrix = torch.cat( (zmatrix, torch.cat( (c[:5].unsqueeze(0), torch.zeros(1, 2).to(device)), dim=1)) , dim=0)
                #f.write("N  %i  %f  %i  %f\n"%(int(c[0]), c[1], int(c[2]), c[3]))
            else:
                zmatrix = torch.cat( (zmatrix, c.unsqueeze(0)) , dim=0)
                #f.write("N  %i  %f  %i  %f  %i  %f\n"%(int(c[0]), c[1], int(c[2]), c[3], int(c[4]), c[5]))
        for cnt, c in enumerate(cwdB):
            if cnt==0:
                zmatrix = torch.cat( (zmatrix, middle[:7].unsqueeze(0)) , dim=0)
                #f.write("N  %i  %f  %i  %f  %i  %f\n"%(int(middle[0]), middle[1], int(middle[2]), middle[3], int(middle[4]), middle[5]))
            elif cnt==1:
                zmatrix = torch.cat( (zmatrix, torch.cat( (c[:3].unsqueeze(0), middle[7:11].unsqueeze(0)), dim=1)) , dim=0)
                #f.write("N  %i  %f  %i  %f  %i  %f\n"%(int(c[0]), c[1], int(middle[6]), middle[7], int(middle[8]), middle[9]))
            elif cnt==2:
                zmatrix = torch.cat( (zmatrix, torch.cat( (c[:5].unsqueeze(0), middle[11:].unsqueeze(0)), dim=1)) , dim=0)
                #f.write("N  %i  %f  %i  %f  %i  %f\n"%(int(c[0]), c[1], int(c[2]), c[3], int(middle[10]), middle[11]))
            else:
                zmatrix = torch.cat( (zmatrix, c.unsqueeze(0)) , dim=0)
                #f.write("N  %i  %f  %i  %f  %i  %f\n"%(int(c[0]), c[1], int(c[2]), c[3], int(c[4]), c[5]))
    
        # now convert into a PDB
        PDB = zmatrix_converter(zmatrix)
        cartesian = PDB.zmatrix_to_cartesian()

