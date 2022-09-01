# Decisions

Documentation about decisions made about the bootstrap and nasateam algorithms.

# Bootstrap

## 6 GHz weather filter

The code we received from Goddard for processing AMSR 2 data includes a 6GHz
vertical channel weather filter.

We have decided NOT to use the 6GHz channel as an additional weather filter for
ASMSR 2 data so as to be consistent across the entire timeseries, including when
using sensors that lack a 6GHz vertical channel.

The Goddard weather filter code that uses `wintrc2` and `wslope2` is located in
`/share/apps/amsr2-cdr/cdr_testdata/bt_amsru_regression/orig_goddard_bt_code/ret_water_amsru2.f`. Originally
set in ret_parameters_amsru2.f.


## Apparent errors in the original Nasa Team code

There we found what we believe are mistakes in the original Nasa Team land
spillover code.

### First bug

While reviewing the NASA Team sea ice concentration code, we found a small
difference between the land spillover methodology between the Northern and
Southern Hemispheres.

In "apply_sst_n.c", there is a loop which adjusts the "minimum ice
concentration" array -- in 10ths of a percent -- based on how far away
from land the ocean pixels are.  Here's the code snippet:

```
/* reset cmin values */

   for(i=0;i<COLS;i++)
   {
     for(j=0;j<ROWS;j++)
     {
        switch(shore.img[j][i])
          {
           case 5:
             if(cmin.img[j][i] > 200) cmin.img[j][i] = 200;
             break;
           case 4:
             if(cmin.img[j][i] > 400) cmin.img[j][i] = 400;
             break;
           case 3:
             if(cmin.img[j][i] > 600) cmin.img[j][i] = *600*;
             break;
           default:
             break;
         }
      }/* ROWS */
   }/* COLS */
```

In the corresponding code for the Southern Hemisphere, "apply_sst_s.c", the
equivalent section has slightly different values for "case 3", which
applies to ocean pixels adjacent to land.  Here is the code snippet from
that source file:

```
/* reset cmin values */


   for(i=0;i<COLS;i++)
   {
     for(j=0;j<ROWS;j++)
     {
        switch(shore.img[j][i])
          {
           case 5:
             if(cmin.img[j][i] > 200) cmin.img[j][i] = 200;
             break;
           case 4:
             if(cmin.img[j][i] > 400) cmin.img[j][i] = 400;
             break;
           case 3:
**********   if(cmin.img[j][i] > 400) cmin.img[j][i] = *400*; **************
             break;
           default:
             break;
         }

     }/* ROWS */
   }/* COLS */
```


(Note that several asterisks were added to the line in question.)

This has the effect of limiting the maximum amout of sea ice concentration
that would be removed during the land spillover algorithm for ocean pixels
adjacent to land.  And this maximum amount is hereby different in the
Southern Hemisphere -- 40% -- than it can be in the Northern Hemisphere --
60%.

We suspect this was an unintentional difference, i.e. that the land spillover
coefficients are not supposed to be different in the Northern and Southern
Hemispheres. For this reason, we have made the behavior consistent between the
two hemispheres unless our correspondence with NASA Goddard determines otherwise.

### Second bug

There appears to be a bug in the Southern Hemisphere (SH) land spillover
code.  A line of code that is corrected in the Northern Hemisphere (NH)
code is not similarly corrected in the Southern Hemisphere code.

The code snippet from the NH code -- apply_sst_n.c -- with the error
described and the corrected code following:


```
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
        /*  This line of code has been edited and exchanged with the line
        directly following as an error is believed to exist in this line.
The
        section of code is included for documentation purposes only and
should not
        be included.  Edit date 10/07/1999.

                -if ice con less than Minimum ice con, ice con = 0-
                if(ice.img[j][i] >= 0 && ice.img[j][i] <= cmin.img[j][i])
                newice.img[j][i] = 0;

        End case documentation. */

/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

           /* if ice con less than Minimum ice con, ice con = 0*/
              if(ice.img[j][i] >= 0 && newice.img [j][i] < 0)
                newice.img[j][i] = 0;
```


The corresponding lines in the SH spillover code -- apply_sst_s.c -- not
corrected and therefore presumably erroneous?:

```
             /* if ice con less than Minimum ice con, ice con = 0 */
               if(ice.img[j][i] >= 0 && ice.img[j][i] <= cmin.img[j][i])
                newice.img[j][i] = 0;
```


We think that the same code should be applied in the SH as in the NH, i.e.:
subtract the thresholded minimum ice concentration value if appropriate, then
ensure that this does not result in an ice concentration less than zero. For
this reason, we have made the behavior consistent between the two hemispheres
unless our correspondence with NASA Goddard determines otherwise.
