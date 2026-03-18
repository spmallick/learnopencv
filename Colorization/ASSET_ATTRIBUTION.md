# Colorization Asset Attribution

This folder includes or references third-party model assets from the original
colorization project by Richard Zhang, Phillip Isola, and Alexei A. Efros:

- Upstream project: <https://github.com/richzhang/colorization>
- Upstream `caffe` branch used for the support files:
  - `pts_in_hull.npy`
  - `models/colorization_deploy_v2.prototxt`
- Original project page referenced by the upstream fetch script:
  - <https://people.eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/>

## Files in This Repo

- `pts_in_hull.npy`
  - Source: `richzhang/colorization` `caffe` branch
  - Original path:
    `<https://raw.githubusercontent.com/richzhang/colorization/caffe/resources/pts_in_hull.npy>`
- `models/colorization_deploy_v2.prototxt`
  - Source: `richzhang/colorization` `caffe` branch
  - Original path:
    `<https://raw.githubusercontent.com/richzhang/colorization/caffe/models/colorization_deploy_v2.prototxt>`

## Release Asset Hosted by LearnOpenCV

- `colorization_release_v2.caffemodel`
  - Hosted as a GitHub release asset in `spmallick/learnopencv` to keep the
    repository checkout small and avoid relying on dead upstream download links.
  - Original model attribution remains with the upstream authors:
    Richard Zhang, Phillip Isola, Alexei A. Efros.
  - SHA-256 of the mirrored asset:
    `f5af1e602646328c792e1094f9876fe9cd4c09ac46fa886e5708a1abc89137b1`

## License

The upstream `richzhang/colorization` repository includes this copyright notice:

`Copyright (c) 2016, Richard Zhang, Phillip Isola, Alexei A. Efros`

Its license permits redistribution and use in source and binary forms, with or
without modification, provided the copyright notice, conditions, and disclaimer
are retained.
