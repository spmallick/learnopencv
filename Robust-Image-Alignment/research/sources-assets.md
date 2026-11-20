# Sample images and ground truth

The companion uses **real camera captures** from two Oxford affine benchmark
sequences, the OpenCV sample collection, and the existing LearnOpenCV document
alignment tutorial. These photographs are not generated examples or synthetic
warps. Any additional digitally warped or degraded variants must be labelled
synthetic separately.

## Exact local layout

Run `python download_assets.py` from any directory. The downloader reads the
adjacent `assets_manifest.json`, downloads immutable Git-commit URLs, checks the
byte count and SHA-256 of every file, and returns `assets/downloads/`.

- `oxford/graf/img1.png` through `img6.png`: graffiti wall photographed under
  increasingly different viewpoints, each 800 × 640 pixels.
- `oxford/graf/H1to2p.xml` through `H1to6p.xml`: benchmark transformations from
  image 1 coordinates into the corresponding target image coordinates.
- `oxford/leuven/img1.png` through `img6.png`: real illumination-change images,
  each 900 × 600 pixels. Cars/building remain at essentially the same viewpoint.
- `oxford/leuven/H1to2p.xml` through `H1to6p.xml`: corresponding benchmark
  transformations, with the same reference-to-moving direction.
- `opencv/box.png` and `box_in_scene.png`: a Biscotti package panel, then that
  panel photographed among several objects. Useful for identifying a known
  printed design from one reference; this does not identify a physical package
  among identical copies.
- `opencv/aloeL.jpg` and `aloeR.jpg`: stereo captures with near plant leaves and
  a farther textured background. Useful for showing why a single homography
  cannot align all depths when the camera translates.
- `opencv/fruits.jpg` and `building.jpg`: target-absent images for the box
  reference. These are easy negatives, not a thorough look-alike rejection set.
- `learnopencv/form.jpg` and `scanned-form.jpg`: the blank form and real photo
  from LearnOpenCV's feature-based alignment article. Handwritten entries are
  real content differences, so intensity differences are not pure registration
  error.

The OpenCV samples also include `graf1.png`, `graf3.png`, `H1to3p.xml`, and
`leuvenA.jpg`/`leuvenB.jpg` for provenance and small familiar examples. Benchmark
evaluation should use the matched PNG sequences and their adjacent matrices,
rather than mix JPEG/sample variants with an assumed transform.

## Benchmark protocol

`assets_manifest.json` contains ten real image pairs across **two sequences**.
This is a compact tutorial experiment, not the full HPatches benchmark and not
evidence for general-purpose performance over arbitrary scenes.

- Validation: `graf_1_2` and `leuven_1_2`.
- Evaluation: `graf_1_3` through `graf_1_6`, and `leuven_1_3` through `leuven_1_6`.
- Each pair includes image paths, the XML filename, a flat nine-value homography
  matrix, capture type, change type, and split.
- Ground-truth matrices map **reference → moving** in original pixel units.
  Invert them if the tested alignment API estimates **moving → reference**.
- Match evaluation and transformation error must account for resizing and
  overlap. Projecting corners outside the common visible region can measure
  extrapolation rather than alignment in observed content; report the chosen
  metric explicitly.
- The box and form examples have no independent corner/homography ground truth
  in this bundle. Use them qualitatively, or create and document independent
  annotations before claiming measured localization accuracy.

## Primary sources and licences

### Oxford affine benchmark

Original source: [Affine Covariant Features](https://www.robots.ox.ac.uk/~vgg/research/affine/).
The primary page describes PPM image packages with homographies and the tested
capture changes (viewpoint, light, blur, rotation/scale and compression).

Citation: Krystian Mikolajczyk, Tinne Tuytelaars, Cordelia Schmid, Andrew
Zisserman, Jiri Matas, Frederik Schaffalitzky, Timor Kadir, and Luc Van Gool,
"A Comparison of Affine Region Detectors," IJCV 65(1/2), 2005, pp. 43–72.

Reproducible mirror:
`opencv/opencv_extra@697e5c3de828f3154e82f222dbe902d622bbe19a`,
`testdata/cv/detectors_descriptors_evaluation/images_datasets/`.

No explicit image redistribution licence was identified on the Oxford source
page or in the `opencv_extra` repository root. Keep these assets as
**download-only third-party data**, retain their original attribution, and do
not represent them as newly Apache- or MIT-licensed project photographs.
The HPatches repository's MIT licence covers its software/documentation and is
not used here as blanket permission for independently sourced imagery.

### OpenCV sample images

Immutable source:
`opencv/opencv@205613a29728c2217049bdddbbe98322b485b34e`, `samples/data/`.
The upstream repository publishes an [Apache 2.0 licence](https://github.com/opencv/opencv/blob/205613a29728c2217049bdddbbe98322b485b34e/LICENSE).
The corresponding LICENSE and COPYRIGHT texts are retained in `assets/`.
Images are fetched unchanged; no new permission for the depicted trademarks or
underlying artwork is asserted.

### LearnOpenCV document example

Primary article: [Feature Based Image Alignment using OpenCV](https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/).
Images are unchanged companion assets from `ImageAlignment-FeatureBased/` at
the full commit recorded in the manifest. The article describes a DMV form and
a mobile-phone photograph of that form. No new licence is asserted for the form
design.

### Related LearnOpenCV reading

- [Image Alignment ECC in OpenCV](https://learnopencv.com/image-alignment-ecc-in-opencv-c-python/)
- [Feature Based Image Alignment using OpenCV](https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/)
- [Introduction to Feature Matching Using Neural Networks](https://learnopencv.com/feature-matching/)

## Visual inspection

`assets/downloads/asset_contact_sheet.jpg` is a local contact sheet for authoring inspection,
generated from downloaded images. It should remain generated/download-only
material alongside the raw third-party data. The selected graffiti image 6
shows a particularly strong viewpoint change; Leuven image 6 shows a strong
illumination change; the aloe pair exposes depth-dependent displacement.
