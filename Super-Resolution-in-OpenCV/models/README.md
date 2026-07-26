# ESPCN x4 model provenance

`ESPCN_x4.pb` is an unchanged pretrained TensorFlow graph from the upstream
[fannymonori/TF-ESPCN](https://github.com/fannymonori/TF-ESPCN) project.

- Upstream revision:
  `a899033b12cd0400454fb5777600883a9d7e86c3`
- Upstream path: `export/ESPCN_x4.pb`
- Expected size: `100323` bytes
- SHA-256:
  `e403f06309229cf36009cd8fb0da032ba7643fae9f15cf94fe562e8edf8fef47`
- Upstream license: Apache License 2.0
- Bundled license text: `TF-ESPCN-LICENSE`

Run `python3 ../download_model.py` from this directory, or
`python3 download_model.py` from the project root, to retrieve the pinned file
and verify its digest before installation. The `.gitignore` intentionally keeps
arbitrary `.pb` files out of Git; release archives may bundle the exact verified
model described above.
