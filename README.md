# patchies

![jeff](assets/jeff.gif)![cat](assets/cat_jeff.gif)

Replace patches of images with other images.

-   breaks up an image into patches
-   for each patch:
    -   look up the approximate nearest neighbour in a dataset of other images
    -   replace the patch with the neighbour image

Uses [nmslib](https://github.com/nmslib/nmslib) for a fast nearest neighbour
lookup.
