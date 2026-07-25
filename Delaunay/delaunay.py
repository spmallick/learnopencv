#!/usr/bin/env python3
"""Render and validate Delaunay triangulation and Voronoi diagrams with OpenCV."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, Sequence

import cv2
import numpy as np


# The bundled regression data is intentionally small enough to inspect visually.
EXPECTED_WIDTH = 512
EXPECTED_HEIGHT = 697
EXPECTED_POINT_RECORDS = 68
EXPECTED_UNIQUE_POINTS = 66
EXPECTED_DUPLICATE_POINTS = 2
EXPECTED_TRIANGLES = 110
EXPECTED_UNIQUE_EDGES = 175
EXPECTED_BOUNDARY_EDGES = 20
EXPECTED_INTERIOR_EDGES = 155
EXPECTED_VORONOI_FACETS = 66
EXPECTED_TRIANGLE_AREA = 30853.0


@dataclass(frozen=True)
class GeometryMetrics:
    """Stable, ordering-independent measurements of the generated geometry."""

    width: int
    height: int
    point_records: int
    unique_points: int
    duplicate_points: int
    triangles: int
    unique_edges: int
    boundary_edges: int
    interior_edges: int
    voronoi_facets: int
    triangle_area: float


def rect_contains(
    rect: tuple[int, int, int, int],
    point: tuple[float, float],
) -> bool:
    """Return whether a point lies inside an OpenCV half-open rectangle."""

    # OpenCV rectangles include their left and top edges but exclude right/bottom.
    x, y, width, height = rect
    # Keep the comparison in floating point because Subdiv2D returns Point2f data.
    point_x, point_y = point
    # Using x + width also makes this correct for rectangles not rooted at (0, 0).
    return (
        x <= point_x < x + width
        and y <= point_y < y + height
    )


def rounded_point(point: Sequence[float]) -> tuple[int, int]:
    """Round a floating-point OpenCV point for raster drawing."""

    # np.rint matches the intent of the official C++ sample's cvRound conversion.
    coordinates = np.rint(np.asarray(point, dtype=np.float32)).astype(np.int32)
    # Python's drawing bindings require ordinary integer coordinates.
    return int(coordinates[0]), int(coordinates[1])


def deterministic_color(index: int) -> tuple[int, int, int]:
    """Return a repeatable BGR color shared with the C++ example."""

    # A simple modular palette avoids random output while keeping adjacent cells distinct.
    blue = 64 + (37 * index) % 192
    green = 64 + (17 * index) % 192
    red = 64 + (29 * index) % 192
    # OpenCV expects colors in blue-green-red order.
    return blue, green, red


def load_points(
    points_path: Path,
    rect: tuple[int, int, int, int],
) -> tuple[list[tuple[float, float]], int]:
    """Read, validate, and de-duplicate landmark records while preserving order."""

    # Fail clearly before OpenCV receives an incomplete or missing point set.
    if not points_path.is_file():
        raise FileNotFoundError(f"Point file not found: {points_path}")

    # Retain insertion order so the visual result remains easy to compare.
    unique_points: list[tuple[float, float]] = []
    # A set makes exact duplicate detection explicit instead of API-dependent.
    seen: set[tuple[float, float]] = set()
    # Track every non-comment record so validation can report skipped duplicates.
    record_count = 0

    # UTF-8 text is sufficient for the two numeric columns used by this example.
    for line_number, raw_line in enumerate(
        points_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        # Permit blank lines and trailing comments in user-supplied point files.
        line = raw_line.split("#", maxsplit=1)[0].strip()
        # Empty or comment-only lines do not represent landmark records.
        if not line:
            continue
        # Each landmark must contain exactly one x and one y coordinate.
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(
                f"{points_path}:{line_number}: expected two coordinates"
            )
        # Parse numerically first so NaN, infinity, and decimal input get clear errors.
        try:
            numeric_point = float(fields[0]), float(fields[1])
        except ValueError as error:
            raise ValueError(
                f"{points_path}:{line_number}: coordinates must be numeric"
            ) from error
        # NaN and infinity can otherwise surface later as opaque OpenCV errors.
        if not np.isfinite(numeric_point).all():
            raise ValueError(
                f"{points_path}:{line_number}: coordinates must be finite"
            )
        # This raster tutorial intentionally accepts integer pixel landmarks only.
        if not all(coordinate.is_integer() for coordinate in numeric_point):
            raise ValueError(
                f"{points_path}:{line_number}: coordinates must be integers"
            )
        # Subdiv2D still receives Point2f values, matching its public API.
        point = float(int(numeric_point[0])), float(int(numeric_point[1]))
        # Subdiv2D raises for points outside its initialization rectangle.
        if not rect_contains(rect, point):
            raise ValueError(
                f"{points_path}:{line_number}: point {point} is outside {rect}"
            )

        # Count the source record even when it repeats an earlier coordinate.
        record_count += 1
        # Skip exact duplicates deterministically; Subdiv2D would reuse the vertex.
        if point in seen:
            continue
        # Remember the coordinate before inserting it into the output sequence.
        seen.add(point)
        # Preserve the first occurrence for predictable insertion and animation.
        unique_points.append(point)

    # A triangulation requires at least three distinct points.
    if len(unique_points) < 3:
        raise ValueError(
            f"{points_path}: at least three unique points are required"
        )
    # Return both unique geometry and the source-record count for transparent metrics.
    return unique_points, record_count


def canonical_triangle(
    triangle: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Convert one Vec6f triangle into an ordering-independent integer tuple."""

    # Reshape six scalar coordinates into three x/y points.
    vertices = np.asarray(triangle, dtype=np.float32).reshape(3, 2)
    # Raster and regression logic use the same explicit rounding rule.
    points = [rounded_point(vertex) for vertex in vertices]
    # Vertex ordering is not an OpenCV API guarantee, so sort it for validation.
    return tuple(sorted(points))


def collect_triangles(
    subdiv: cv2.Subdiv2D,
    rect: tuple[int, int, int, int],
) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    """Collect in-bounds triangles without relying on OpenCV output ordering."""

    # Python returns a floating-point N-by-6 representation of Vec6f triangles.
    raw_triangles = np.asarray(subdiv.getTriangleList(), dtype=np.float32)
    # Normalize both empty and populated binding representations.
    raw_triangles = raw_triangles.reshape(-1, 6)
    # A set prevents accidental duplicate triangles from affecting validation.
    triangles: set[
        tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    ] = set()

    # Inspect each triangle independently because collection order is unspecified.
    for raw_triangle in raw_triangles:
        # Check the original Point2f values against half-open image bounds.
        float_vertices = raw_triangle.reshape(3, 2)
        if not all(
            rect_contains(rect, (float(vertex[0]), float(vertex[1])))
            for vertex in float_vertices
        ):
            continue
        # Canonicalization makes subsequent edge and area metrics order-independent.
        triangles.add(canonical_triangle(raw_triangle))

    # Sorting is only for deterministic iteration and drawing, not geometric meaning.
    return sorted(triangles)


def collect_voronoi(
    subdiv: cv2.Subdiv2D,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Return all nonvirtual Voronoi facets and their rounded centers."""

    # Passing an empty index list asks OpenCV for every nonvirtual facet.
    raw_facets, raw_centers = subdiv.getVoronoiFacetList([])
    # Centers correspond to inserted sites, but their order is unspecified.
    centers_array = np.asarray(raw_centers, dtype=np.float32).reshape(-1, 2)
    # A mismatch would make facet-to-site rendering ambiguous.
    if len(raw_facets) != len(centers_array):
        raise RuntimeError("OpenCV returned mismatched Voronoi facets and centers")
    # Pair every normalized facet with its rounded integer site.
    cells = [
        (
            np.asarray(facet, dtype=np.float32).reshape(-1, 2),
            rounded_point(center),
        )
        for facet, center in zip(raw_facets, centers_array)
    ]
    # Sort by site so palette assignment never depends on API collection order.
    cells.sort(key=lambda cell: cell[1])
    # Unzip the canonical sequence while keeping every facet with its center.
    facets = [cell[0] for cell in cells]
    centers = [cell[1] for cell in cells]
    # Return a deterministic cell sequence shared with the C++ implementation.
    return facets, centers


def draw_point(
    image: np.ndarray,
    point: tuple[float, float],
    color: tuple[int, int, int],
) -> None:
    """Draw one landmark with modern OpenCV constants."""

    # cv2.circle requires an integer center in current Python bindings.
    center = rounded_point(point)
    # FILLED creates a solid marker and LINE_AA keeps its edge smooth.
    cv2.circle(image, center, 2, color, cv2.FILLED, cv2.LINE_AA)


def draw_delaunay(
    image: np.ndarray,
    triangles: Iterable[
        tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    ],
    color: tuple[int, int, int],
) -> None:
    """Draw every canonical Delaunay triangle on an image."""

    # Each triangle contributes three line segments.
    for point_1, point_2, point_3 in triangles:
        # Draw the first edge with a one-pixel antialiased stroke.
        cv2.line(image, point_1, point_2, color, 1, cv2.LINE_AA)
        # Draw the second edge using the same rasterization settings.
        cv2.line(image, point_2, point_3, color, 1, cv2.LINE_AA)
        # Close the triangle with its final edge.
        cv2.line(image, point_3, point_1, color, 1, cv2.LINE_AA)


def draw_voronoi(
    image: np.ndarray,
    facets: Sequence[np.ndarray],
    centers: Sequence[tuple[int, int]],
) -> None:
    """Fill Voronoi cells with deterministic colors and mark their sites."""

    # OpenCV returns facets and centers as corresponding sequences.
    if len(facets) != len(centers):
        raise RuntimeError("OpenCV returned mismatched Voronoi facets and centers")

    # Pair each cell with its site while assigning a repeatable palette index.
    for index, (facet, center) in enumerate(zip(facets, centers)):
        # A valid Voronoi polygon needs at least three vertices.
        if len(facet) < 3:
            raise RuntimeError(f"Voronoi facet {index} has fewer than 3 vertices")
        # Polygon drawing requires an explicit int32 array in Python.
        polygon = np.rint(facet).astype(np.int32).reshape(-1, 1, 2)
        # Fill the cell before drawing its black outline and center marker.
        cv2.fillConvexPoly(
            image,
            polygon,
            deterministic_color(index),
            cv2.LINE_AA,
        )
        # The outline makes neighboring cells distinguishable at similar colors.
        cv2.polylines(
            image,
            [polygon],
            True,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        # Mark the generating site at the center returned by Subdiv2D.
        cv2.circle(
            image,
            center,
            3,
            (0, 0, 0),
            cv2.FILLED,
            cv2.LINE_AA,
        )


def triangle_area(
    triangle: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> float:
    """Calculate the unsigned area of one integer-coordinate triangle."""

    # Unpack the vertices for the two-dimensional shoelace formula.
    (x_1, y_1), (x_2, y_2), (x_3, y_3) = triangle
    # The determinant is twice the signed triangle area.
    twice_area = (
        x_1 * (y_2 - y_3)
        + x_2 * (y_3 - y_1)
        + x_3 * (y_1 - y_2)
    )
    # Geometry validation uses the unsigned area in pixel-coordinate units.
    return abs(twice_area) / 2.0


def edge_counts(
    triangles: Sequence[
        tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    ],
) -> tuple[int, int, int]:
    """Count unique, hull, and interior edges independently of triangle order."""

    # Each normalized edge maps to the number of incident triangles.
    incidences: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
    # Visit every triangle without assuming a clockwise vertex order.
    for triangle in triangles:
        # A triangle has the three undirected edges below.
        edges = (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        )
        # Canonicalize each endpoint pair before incrementing its incidence.
        for edge in edges:
            normalized = tuple(sorted(edge))
            incidences[normalized] = incidences.get(normalized, 0) + 1

    # Hull edges belong to exactly one triangle.
    boundary_edges = sum(count == 1 for count in incidences.values())
    # Interior edges are shared by exactly two triangles.
    interior_edges = sum(count == 2 for count in incidences.values())
    # Any other incidence indicates duplicate or nonmanifold triangulation data.
    invalid_edges = [
        edge for edge, count in incidences.items() if count not in (1, 2)
    ]
    if invalid_edges:
        raise RuntimeError(
            f"Triangulation contains {len(invalid_edges)} invalid edge incidences"
        )
    # Return the three stable counts used by the regression contract.
    return len(incidences), boundary_edges, interior_edges


def calculate_metrics(
    image: np.ndarray,
    point_records: int,
    points: Sequence[tuple[float, float]],
    triangles: Sequence[
        tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    ],
    facets: Sequence[np.ndarray],
) -> GeometryMetrics:
    """Calculate stable geometry measurements for reporting and validation."""

    # Image height and width define the Subdiv2D working rectangle.
    height, width = image.shape[:2]
    # Count edges through topology rather than relying on triangle list ordering.
    unique_edges, boundary_edges, interior_edges = edge_counts(triangles)
    # Sum triangle areas to compare the triangulation with its convex hull.
    total_area = sum(triangle_area(triangle) for triangle in triangles)
    # Construct the immutable result after all consistency checks pass.
    return GeometryMetrics(
        width=width,
        height=height,
        point_records=point_records,
        unique_points=len(points),
        duplicate_points=point_records - len(points),
        triangles=len(triangles),
        unique_edges=unique_edges,
        boundary_edges=boundary_edges,
        interior_edges=interior_edges,
        voronoi_facets=len(facets),
        triangle_area=total_area,
    )


def validate_metrics(
    metrics: GeometryMetrics,
    points: Sequence[tuple[float, float]],
    triangles: Sequence[
        tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    ],
    centers: Sequence[tuple[int, int]],
) -> None:
    """Check the bundled example's ordering-independent regression contract."""

    # Rounded membership checks are appropriate for the bundled integer landmarks.
    rounded_sites = {rounded_point(point) for point in points}
    # Every nonvirtual Voronoi center should be one of the unique bundled sites.
    if set(centers) != rounded_sites:
        raise RuntimeError("Validation failed: Voronoi centers differ from sites")
    # Every Delaunay vertex should also correspond to one bundled site.
    if any(vertex not in rounded_sites for triangle in triangles for vertex in triangle):
        raise RuntimeError("Validation failed: triangle has a non-site vertex")
    # The bundled integer landmarks should never produce a degenerate raster triangle.
    if any(triangle_area(triangle) <= 0 for triangle in triangles):
        raise RuntimeError("Validation failed: triangle has nonpositive area")

    # Keep the expected values together so failures identify the precise invariant.
    expected = GeometryMetrics(
        width=EXPECTED_WIDTH,
        height=EXPECTED_HEIGHT,
        point_records=EXPECTED_POINT_RECORDS,
        unique_points=EXPECTED_UNIQUE_POINTS,
        duplicate_points=EXPECTED_DUPLICATE_POINTS,
        triangles=EXPECTED_TRIANGLES,
        unique_edges=EXPECTED_UNIQUE_EDGES,
        boundary_edges=EXPECTED_BOUNDARY_EDGES,
        interior_edges=EXPECTED_INTERIOR_EDGES,
        voronoi_facets=EXPECTED_VORONOI_FACETS,
        triangle_area=EXPECTED_TRIANGLE_AREA,
    )
    # Dataclass equality gives an exact comparison for these integer/half-pixel metrics.
    if metrics != expected:
        raise RuntimeError(f"Validation failed: expected {expected}, got {metrics}")
    # Euler's formula provides an independent planar-triangulation topology check.
    faces_including_exterior = metrics.triangles + 1
    if metrics.unique_points - metrics.unique_edges + faces_including_exterior != 2:
        raise RuntimeError("Validation failed: Euler characteristic is not 2")


def validate_rendered_images(
    source: np.ndarray,
    delaunay_image: np.ndarray,
    voronoi_image: np.ndarray,
) -> None:
    """Confirm validation generated meaningful visual content, not blank files."""

    # Removing all triangle/point drawing must make the regression fail.
    if np.array_equal(delaunay_image, source):
        raise RuntimeError("Validation failed: Delaunay output equals the input")
    # A blank Voronoi canvas indicates that facet drawing did not run.
    if not np.any(voronoi_image):
        raise RuntimeError("Validation failed: Voronoi output is blank")
    # Multiple distinct BGR colors prove that more than one flat fill was rendered.
    unique_colors = np.unique(voronoi_image.reshape(-1, 3), axis=0)
    if len(unique_colors) < 3:
        raise RuntimeError("Validation failed: Voronoi output lacks color variation")


def write_image(path: Path, image: np.ndarray) -> None:
    """Write an output image and fail if the encoder or destination rejects it."""

    # cv2.imwrite returns False instead of raising for several write failures.
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write output image: {path}")
    # Decode the file again so validation covers the actual artifact on disk.
    decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if decoded is None or decoded.size == 0:
        raise OSError(f"Could not read generated output image: {path}")
    # Output dimensions and channel layout must match the rendered array.
    if decoded.shape != image.shape:
        raise OSError(
            f"Generated output shape mismatch for {path}: "
            f"{decoded.shape} != {image.shape}"
        )


def ensure_output_paths_are_safe(
    image_path: Path,
    points_path: Path,
    output_dir: Path,
) -> None:
    """Reject output paths that resolve to either input file."""

    # resolve(strict=False) normalizes missing outputs and follows existing symlinks.
    input_paths = {
        image_path.expanduser().resolve(),
        points_path.expanduser().resolve(),
    }
    # Both documented output filenames must be checked before either is written.
    output_paths = (
        (output_dir / "delaunay.png").expanduser().resolve(),
        (output_dir / "voronoi.png").expanduser().resolve(),
    )
    # Stop before mkdir or imwrite can replace a source image or landmark file.
    for output_path in output_paths:
        for input_path in input_paths:
            # Normalized equality handles the ordinary and symlinked path cases.
            paths_match = output_path == input_path
            # samefile additionally detects existing hard links and case aliases.
            if not paths_match and output_path.exists() and input_path.exists():
                paths_match = output_path.samefile(input_path)
            if paths_match:
                raise ValueError(
                    f"Output path would overwrite an input file: {output_path}"
                )


def build_subdivision(
    rect: tuple[int, int, int, int],
    points: Sequence[tuple[float, float]],
    image: np.ndarray,
    display: bool,
    animate: bool,
    animation_delay_ms: int,
) -> cv2.Subdiv2D:
    """Insert points and optionally display progressive triangulation."""

    # Construct the subdivision over the exact image rectangle.
    subdiv = cv2.Subdiv2D(rect)
    # Insert each unique site once in its source order.
    for point in points:
        # Subdiv2D returns an existing vertex ID for a duplicate, already filtered above.
        subdiv.insert(point)
        # Animation is deliberately opt-in so automated and remote runs never block.
        if display and animate:
            # Render the partial triangulation on a fresh image copy.
            frame = image.copy()
            # Canonical collection keeps the animation independent of API ordering.
            partial_triangles = collect_triangles(subdiv, rect)
            # White lines remain visible against the portrait.
            draw_delaunay(frame, partial_triangles, (255, 255, 255))
            # Display the progressive result in the tutorial window.
            cv2.imshow("Delaunay Triangulation", frame)
            # A positive delay lets the window process events between insertions.
            cv2.waitKey(animation_delay_ms)
    # Return the completed subdivision for final geometry and Voronoi extraction.
    return subdiv


def run(
    image_path: Path,
    points_path: Path,
    output_dir: Path,
    display: bool = False,
    animate: bool = False,
    animation_delay_ms: int = 100,
    validate: bool = False,
) -> GeometryMetrics:
    """Run the complete Delaunay/Voronoi example and return its metrics."""

    # Resolve every intended artifact before reading or writing user-controlled paths.
    ensure_output_paths_are_safe(image_path, points_path, output_dir)
    # Read the color image without depending on the caller's current directory.
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise FileNotFoundError(f"Could not read input image: {image_path}")
    # This visualization expects an ordinary three-channel BGR input.
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected a three-channel image: {image_path}")

    # Build the OpenCV half-open rectangle from the decoded image dimensions.
    height, width = image.shape[:2]
    rect = (0, 0, width, height)
    # Parse and explicitly de-duplicate the landmark records.
    points, point_records = load_points(points_path, rect)
    # Insert the sites, optionally rendering the progressive triangulation.
    subdiv = build_subdivision(
        rect,
        points,
        image,
        display,
        animate,
        animation_delay_ms,
    )
    # Extract canonical geometry only after every site has been inserted.
    triangles = collect_triangles(subdiv, rect)
    # Extract all nonvirtual Voronoi facets and their generating sites.
    facets, centers = collect_voronoi(subdiv)

    # Draw the final triangulation on a copy of the original image.
    delaunay_image = image.copy()
    draw_delaunay(delaunay_image, triangles, (255, 255, 255))
    # Overlay the unique landmark sites in red.
    for point in points:
        draw_point(delaunay_image, point, (0, 0, 255))

    # Start the Voronoi visualization from a black canvas of the same size/type.
    voronoi_image = np.zeros_like(image)
    # Fill and outline every returned Voronoi facet deterministically.
    draw_voronoi(voronoi_image, facets, centers)

    # Calculate validation metrics before writing or displaying outputs.
    metrics = calculate_metrics(
        image,
        point_records,
        points,
        triangles,
        facets,
    )
    # The validation flag activates the bundled-data regression contract.
    if validate:
        validate_metrics(metrics, points, triangles, centers)
        validate_rendered_images(image, delaunay_image, voronoi_image)

    # Create the output directory explicitly for reliable CLI behavior.
    output_dir.mkdir(parents=True, exist_ok=True)
    # Lossless PNG output avoids codec-dependent JPEG artifacts.
    write_image(output_dir / "delaunay.png", delaunay_image)
    # Verify the Voronoi artifact through the same write/read path.
    write_image(output_dir / "voronoi.png", voronoi_image)

    # Display remains optional because headless environments may have no window backend.
    if display:
        # Show the completed triangulation.
        cv2.imshow("Delaunay Triangulation", delaunay_image)
        # Show its dual Voronoi diagram beside it.
        cv2.imshow("Voronoi Diagram", voronoi_image)
        # Wait for one key before closing both windows.
        cv2.waitKey(0)
        # Release GUI resources cleanly after the user dismisses the example.
        cv2.destroyAllWindows()

    # Return the metrics so tests exercise the real implementation.
    return metrics


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface shared by interactive and test runs."""

    # Resolve bundled assets relative to this file, never the working directory.
    source_dir = Path(__file__).resolve().parent
    # A focused description keeps --help useful for tutorial readers.
    parser = argparse.ArgumentParser(
        description="Render Delaunay triangulation and Voronoi diagrams."
    )
    # Permit callers to replace the bundled portrait.
    parser.add_argument(
        "--image",
        type=Path,
        default=source_dir / "obama.jpg",
        help="input image (default: bundled obama.jpg)",
    )
    # Permit callers to replace the bundled x/y landmark file.
    parser.add_argument(
        "--points",
        type=Path,
        default=source_dir / "obama.txt",
        help="two-column landmark file (default: bundled obama.txt)",
    )
    # Keep generated artifacts out of the tracked source directory by default.
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=source_dir / "output",
        help="directory for delaunay.png and voronoi.png",
    )
    # Provide both positive and negative display flags for explicit automation.
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument(
        "--display",
        dest="display",
        action="store_true",
        help="open interactive result windows",
    )
    display_group.add_argument(
        "--no-display",
        dest="display",
        action="store_false",
        help="do not open GUI windows (default)",
    )
    # Headless execution is the safe default on servers and in CI.
    parser.set_defaults(display=False)
    # Progressive insertion is useful pedagogically but intentionally opt-in.
    animation_group = parser.add_mutually_exclusive_group()
    animation_group.add_argument(
        "--animate",
        dest="animate",
        action="store_true",
        help="animate point insertion when --display is enabled",
    )
    animation_group.add_argument(
        "--no-animation",
        dest="animate",
        action="store_false",
        help="disable insertion animation (default)",
    )
    # Static output is the reproducible default.
    parser.set_defaults(animate=False)
    # Expose the animation delay without hardcoding it in the rendering loop.
    parser.add_argument(
        "--animation-delay-ms",
        type=int,
        default=100,
        help="delay between animated insertions in milliseconds",
    )
    # Validation checks stable topology rather than undocumented output ordering.
    parser.add_argument(
        "--validate",
        action="store_true",
        help="check the bundled 512x697 regression data",
    )
    # Return the configured parser to both main and tests.
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments, run the example, and report concise metrics."""

    # Build the parser once so validation errors use consistent CLI formatting.
    parser = build_parser()
    # Parse an injected sequence in tests or sys.argv in normal execution.
    args = parser.parse_args(argv)
    # Animation without a display would do invisible work and usually signals a typo.
    if args.animate and not args.display:
        parser.error("--animate requires --display")
    # OpenCV waitKey expects a positive delay for progressive animation.
    if args.animation_delay_ms <= 0:
        parser.error("--animation-delay-ms must be positive")

    # Convert runtime errors into a controlled nonzero CLI result.
    try:
        metrics = run(
            image_path=args.image,
            points_path=args.points,
            output_dir=args.output_dir,
            display=args.display,
            animate=args.animate,
            animation_delay_ms=args.animation_delay_ms,
            validate=args.validate,
        )
    except (cv2.error, OSError, RuntimeError, ValueError) as error:
        # Print no traceback for expected input, output, or validation failures.
        print(f"error: {error}", file=sys.stderr)
        # Exit code two denotes invalid input or an unusable runtime environment.
        return 2

    # Report the exact library under test before the geometry summary.
    print(f"OpenCV version: {cv2.__version__}")
    # Keep the summary compact enough for both readers and CTest logs.
    print(
        "Geometry: "
        f"{metrics.point_records} records, "
        f"{metrics.unique_points} unique points, "
        f"{metrics.triangles} triangles, "
        f"{metrics.voronoi_facets} Voronoi facets"
    )
    # Emit the success marker only after geometry and output verification pass.
    if args.validate:
        print("DELAUNAY_VALIDATION_OK")
    # Zero signals successful rendering and validation.
    return 0


if __name__ == "__main__":
    # Propagate the explicit CLI status to the shell and test harness.
    raise SystemExit(main())
