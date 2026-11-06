The examples have strong rib detail, a bright central mediastinum, and
variable black padding. Several pneumonia-labelled images show broader
gray haze or patchy brightness, while many normal-labelled images have
darker spaces between ribs. There is substantial overlap: brightness
alone would be unreliable. I therefore combine intensity bands, contrast
at two scales, and a modest penalty for sharp rib-like detail.

The conv1 channels are:

  ----------------------------------------------------------------------------
  **Channels**   **Function**
  -------------- -------------------------------------------------------------
  0              Weighted 5 × 5 average brightness.

  1              Its complement, allowing max pooling to preserve information
                 about dark regions.

  2              Constant one; supports contrast calculations without
                 introducing artificial contrast from padding this constant.

  3              A narrower, embedded 3 × 3 average.

  4--5           Positive and negative horizontal intensity gradients.

  6--7           Positive and negative vertical intensity gradients.

  8--9           Positive and negative down-right diagonal gradients.

  10--11         Positive and negative up-right diagonal gradients.

  12--13         Bright and dark vertical ridges.

  14--15         Bright and dark horizontal ridges.
  ----------------------------------------------------------------------------

After the first pooling operation, channels 0 and 1 recover a local
brightness midpoint and range. Conv2 combines these with the other
features:

  -----------------------------------------------------------------------------
  **Channels**   **Combination**                             **Final
                                                             coefficient ×
                                                             spatial mask**
  -------------- ------------------------------------------- ------------------
  0--1           Smoothed midpoint brightness and its        Separate
                 complement                                  comparison below

  2--3           Darkness hinges at 0.55 and 0.275; together −3, +6
                 form a soft air-space intensity band that   
                 vanishes at black                           

  4--6           Brightness hinges at 0.25, 0.50, and 0.75;  +3.5, −3.5, −3.5
                 together favor intermediate brightness and  
                 suppress pure white                         

  7              Local brightness range                      −0.50

  8--15          Broader positive and negative gradients in  +0.30 each
                 four directions                             

  16--19         Fine gradient magnitude in each direction   −0.25 each

  20--21         Positive and negative differences between   +0.40, −0.20
                 narrow- and wide-blur maxima                

  22--25         Averaged vertical and horizontal ridge      −0.075 each
                 responses, both signs                       

  26--27         Broad bright and dark center-surround       +1.50, −1.00
                 contrast                                    

  28--29         Fine contrast with a corresponding ridge    +0.75, −0.35
                 penalty                                     

  30--31         Broad contrast with a sharp-edge penalty    +3.00, −2.00
  -----------------------------------------------------------------------------

These are approximate image features, not anatomical detectors. The
weighting favors broad opacity-like patterns and darker air spaces as
opposing evidence; sharp detail receives weaker negative weights.

For channels 2--31, the spatial mask is the following matrix divided by
its sum, 5.10:

\\\[ L=\\frac{1}{5.10} \\begin{bmatrix} 0&0&0&0\\\\ 0.35&1&1&0.35\\\\
0.45&1&0.50&0.45\\\\ 0&0&0&0 \\end{bmatrix}. \\\]

This concentrates on the middle chest, reduces lateral background
contributions, and downweights cell (2,2), which often contains
substantial heart shadow. The top and bottom rows receive zero weight
because padding, shoulders, and diaphragm occupy much of them.

Channel 0 receives \\(0.20(L-R)\\), and channel 1 receives its negative,
where \\(R\\) is one at (2,2) and zero elsewhere. This adds a weak
comparison with the lower central chest; uniform additive brightness
shifts cancel in this term.

I set the bias to **−0.205**. For an ideal constant interior at
brightness 0.42, the intensity-band contribution is
\\(3.5(0.42-0.25)-3(0.55-0.42)=0.205\\). This supplies an assumed
crossover point, not a calibrated probability threshold.

My untested prediction is **approximately 0.60 AUROC on similar held-out
images and 0.55 at another hospital**, with substantial uncertainty.
