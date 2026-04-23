#!/usr/bin/env python3
"""
Section Properties Calculator
Imports geometric shapes from DXF and calculates comprehensive section properties.

Features:
1. Import geometry from DXF (polyline, arc, line)
2. Recognize hollow and solid sections based on containment
3. Calculate section properties: area, perimeter, centroid, moments of inertia,
   polar moment, neutral axis distances, radius of gyration, elastic and plastic moduli
"""

import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import ezdxf
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union, triangulate
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors
import io
import os


@dataclass
class SectionProperties:
    """Container for all calculated section properties"""
    area: float
    perimeter: float
    centroid: Tuple[float, float]
    Ixx_c: float  # Moment of inertia about centroidal x-axis
    Iyy_c: float  # Moment of inertia about centroidal y-axis
    Ixy_c: float  # Product of inertia about centroidal axes
    principal_I_major: float  # Major principal moment of inertia
    principal_I_minor: float  # Minor principal moment of inertia
    principal_angle_deg: float  # Angle of major principal axis from x-axis (CCW)
    polar_J_c: float  # Polar moment of inertia
    c_major_pos: float  # Distance to extreme fiber (positive major axis)
    c_major_neg: float  # Distance to extreme fiber (negative major axis)
    c_minor_pos: float  # Distance to extreme fiber (positive minor axis)
    c_minor_neg: float  # Distance to extreme fiber (negative minor axis)
    S_major_pos: float  # Elastic section modulus (positive major axis)
    S_major_neg: float  # Elastic section modulus (negative major axis)
    S_minor_pos: float  # Elastic section modulus (positive minor axis)
    S_minor_neg: float  # Elastic section modulus (negative minor axis)
    r_gyration_major: float  # Radius of gyration (major axis)
    r_gyration_minor: float  # Radius of gyration (minor axis)
    Z_plastic_major: float  # Plastic section modulus (major axis)
    Z_plastic_minor: float  # Plastic section modulus (minor axis)


# ==================== GEOMETRY UTILITIES ====================

def polygon_moments(vertices: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """
    Compute signed area, centroid, and second moments about origin for a simple polygon.
    Uses Green's theorem formulas.
    
    Returns: (A, Cx, Cy, Ixx, Iyy, Ixy)
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    
    # Close polygon by rolling
    x2 = np.roll(x, -1)  # Shifts array left by 1, wrapping last to first
    y2 = np.roll(y, -1)  # This "closes" the polygon for calculation
    cross = x * y2 - x2 * y
    
    A = 0.5 * np.sum(cross)
    
    if abs(A) < 1e-12:
        raise ValueError("Degenerate polygon with near-zero area")
    
    Cx = (1 / (6 * A)) * np.sum((x + x2) * cross)
    Cy = (1 / (6 * A)) * np.sum((y + y2) * cross)
    
    Ixx = (1 / 12) * np.sum((y**2 + y * y2 + y2**2) * cross)
    Iyy = (1 / 12) * np.sum((x**2 + x * x2 + x2**2) * cross)
    Ixy = (1 / 24) * np.sum((x * y2 + 2 * x * y + 2 * x2 * y2 + x2 * y) * cross)
    
    return A, Cx, Cy, Ixx, Iyy, Ixy


def composite_polygon_properties(poly: Polygon) -> Tuple[float, float, Tuple[float, float], float, float, float]:
    """
    Compute properties for polygon with holes (hollow sections).
    Returns: (area, perimeter, centroid, Ixx, Iyy, Ixy) about origin
    """
    # Process exterior - ensure CCW orientation
    ext = np.array(poly.exterior.coords)[:-1]
    A_e, Cx_e, Cy_e, Ixx_e, Iyy_e, Ixy_e = polygon_moments(ext)
    
    # If exterior is CW (negative area), reverse it
    if A_e < 0:
        ext = ext[::-1]
        A_e, Cx_e, Cy_e, Ixx_e, Iyy_e, Ixy_e = polygon_moments(ext)
    
    # Initialize with exterior values
    A = A_e
    Cx = Cx_e * A_e
    Cy = Cy_e * A_e
    Ixx = Ixx_e
    Iyy = Iyy_e
    Ixy = Ixy_e
    
    # Process holes (interiors) - ensure CW orientation for proper subtraction
    for ring in poly.interiors:
        hole = np.array(ring.coords)[:-1]
        A_h, Cx_h, Cy_h, Ixx_h, Iyy_h, Ixy_h = polygon_moments(hole)
        
        # If hole is CCW (positive), reverse it to be CW (negative)
        if A_h > 0:
            hole = hole[::-1]
            A_h, Cx_h, Cy_h, Ixx_h, Iyy_h, Ixy_h = polygon_moments(hole)
        
        # Add signed contributions (negative for holes)
        A += A_h
        Cx += Cx_h * A_h
        Cy += Cy_h * A_h
        Ixx += Ixx_h
        Iyy += Iyy_h
        Ixy += Ixy_h
    
    if A <= 0:
        raise ValueError(f"Composite area is non-positive: {A}")
    
    Cx /= A
    Cy /= A
    
    return abs(A), abs(poly.length), (Cx, Cy), Ixx, Iyy, Ixy


def translate_moments_to_centroid(Ixx_o: float, Iyy_o: float, Ixy_o: float, 
                                   A: float, Cx: float, Cy: float) -> Tuple[float, float, float]:
    """
    Apply parallel axis theorem to move moments from origin to centroid.
    """
    Ixx_c = Ixx_o - A * Cy**2
    Iyy_c = Iyy_o - A * Cx**2
    Ixy_c = Ixy_o - A * Cx * Cy
    return Ixx_c, Iyy_c, Ixy_c


def principal_moments_and_angle(Ixx_c: float, Iyy_c: float, Ixy_c: float) -> Tuple[float, float, float]:
    """
    Compute principal moments of inertia and orientation angle.
    Returns: (I_major, I_minor, angle_deg)
    """
    I_avg = 0.5 * (Ixx_c + Iyy_c)
    R = math.sqrt(((Ixx_c - Iyy_c) / 2)**2 + Ixy_c**2)
    
    I_major = I_avg + R
    I_minor = I_avg - R
    
    # Angle where product of inertia is zero
    theta = 0.5 * math.atan2(-2 * Ixy_c, Iyy_c - Ixx_c)
    angle_deg = math.degrees(theta)
    
    return I_major, I_minor, angle_deg


def rotate_points(points: np.ndarray, angle_rad: float, origin: Tuple[float, float]) -> np.ndarray:
    """
    Rotate points by angle_rad around origin.
    """
    R = np.array([[math.cos(angle_rad), -math.sin(angle_rad)],
                  [math.sin(angle_rad),  math.cos(angle_rad)]])
    shifted = points - np.array(origin)
    return shifted @ R.T


def polygon_extents_along_axes(poly: Polygon, centroid: Tuple[float, float], 
                               angle_rad: float) -> Tuple[float, float, float, float]:
    """
    Compute extreme distances along principal axes from centroid.
    Returns: (c_major_pos, c_major_neg, c_minor_pos, c_minor_neg)
    """
    # Collect all boundary points (exterior + holes)
    coords = list(poly.exterior.coords)[:-1]
    for ring in poly.interiors:
        coords += list(ring.coords)[:-1]
    
    pts = np.array(coords)
    pts_rot = rotate_points(pts, angle_rad, centroid)
    
    xprime = pts_rot[:, 0]
    yprime = pts_rot[:, 1]
    
    c_major_pos = np.max(xprime)
    c_major_neg = -np.min(xprime)
    c_minor_pos = np.max(yprime)
    c_minor_neg = -np.min(yprime)
    
    return c_major_pos, c_major_neg, c_minor_pos, c_minor_neg


# ==================== PLASTIC MODULUS CALCULATION ====================

def neutral_axis_position_for_equal_area(tris: List[Polygon], axis: str, angle_rad: float, 
                                        centroid: Tuple[float, float], bounds: Tuple[float, float]) -> float:
    """
    Find neutral axis position where areas on either side are equal (for plastic analysis).
    """
    def area_difference(t: float) -> float:
        diff = 0.0
        for tri in tris:
            c = np.array(tri.centroid.coords[0])
            c_rot = rotate_points(c.reshape(1, 2), angle_rad, centroid)[0]
            coord = c_rot[0] if axis == 'major' else c_rot[1]
            a = tri.area
            if coord > t:
                diff += a
            else:
                diff -= a
        return diff
    
    t_low, t_high = bounds
    f_low = area_difference(t_low)
    f_high = area_difference(t_high)
    
    # Ensure root is in interval
    for _ in range(30):
        if f_low * f_high <= 0:
            break
        span = t_high - t_low
        t_low -= span
        t_high += span
        f_low = area_difference(t_low)
        f_high = area_difference(t_high)
    
    # Bisection method
    for _ in range(60):
        t_mid = 0.5 * (t_low + t_high)
        f_mid = area_difference(t_mid)
        if abs(f_mid) < 1e-9:
            return t_mid
        if f_low * f_mid <= 0:
            t_high = t_mid
            f_high = f_mid
        else:
            t_low = t_mid
            f_low = f_mid
    
    return 0.5 * (t_low + t_high)


def plastic_section_modulus(tris: List[Polygon], axis: str, angle_rad: float, 
                            centroid: Tuple[float, float], bounds: Tuple[float, float]) -> float:
    """
    Approximate plastic section modulus using triangulation.
    """
    t_NA = neutral_axis_position_for_equal_area(tris, axis, angle_rad, centroid, bounds)
    
    Z = 0.0
    for tri in tris:
        c = np.array(tri.centroid.coords[0])
        c_rot = rotate_points(c.reshape(1, 2), angle_rad, centroid)[0]
        coord = c_rot[0] if axis == 'major' else c_rot[1]
        a = tri.area
        d = abs(coord - t_NA)
        Z += a * d
    
    return Z


# ==================== DXF IMPORT UTILITIES ====================

def closed_lwpolyline_to_linestring(entity) -> Optional[LineString]:
    """Convert closed LWPOLYLINE to LineString."""
    try:
        if entity.dxf.closed:
            pts = [(p[0], p[1]) for p in entity.get_points('xy')]
            return LineString(pts + [pts[0]])
        return None
    except Exception:
        return None


def polyline_to_linestring(entity) -> Optional[LineString]:
    """Convert closed POLYLINE to LineString."""
    try:
        pts = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
        if len(pts) < 3:
            return None
        
        # Check if closed or if endpoints match
        tolerance = 1e-6
        is_closed = entity.is_closed or math.dist(pts[0], pts[-1]) < tolerance
        
        if is_closed:
            if math.dist(pts[0], pts[-1]) < tolerance:
                return LineString(pts)
            else:
                return LineString(pts + [pts[0]])
        return None
    except Exception:
        return None


def circle_to_polygon(entity, num_points: int = 64) -> Optional[Polygon]:
    """Convert CIRCLE entity to polygon approximation."""
    try:
        cx = entity.dxf.center.x
        cy = entity.dxf.center.y
        r = entity.dxf.radius
        
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        pts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]
        return Polygon(pts)
    except Exception:
        return None


def arc_to_linestring(entity, num_points: int = 32) -> Optional[LineString]:
    """Convert ARC entity to LineString approximation."""
    try:
        cx = entity.dxf.center.x
        cy = entity.dxf.center.y
        r = entity.dxf.radius
        start_angle = math.radians(entity.dxf.start_angle)
        end_angle = math.radians(entity.dxf.end_angle)
        
        # Handle angle wrapping
        if end_angle <= start_angle:
            end_angle += 2 * np.pi
        
        angles = np.linspace(start_angle, end_angle, num_points)
        pts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]
        return LineString(pts)
    except Exception:
        return None


def connect_entities_to_polygons(entities: List) -> List[LineString]:
    """
    Connect LINE and ARC entities that form closed loops.
    Returns list of closed LineStrings.
    """
    if not entities:
        return []
    
    # Extract endpoints and interpolated points from each entity
    segments = []
    for entity in entities:
        try:
            if entity.dxftype() == 'LINE':
                start = (entity.dxf.start.x, entity.dxf.start.y)
                end = (entity.dxf.end.x, entity.dxf.end.y)
                segments.append((start, end, [start, end]))
            elif entity.dxftype() == 'ARC':
                ls = arc_to_linestring(entity)
                if ls and len(ls.coords) >= 2:
                    pts = list(ls.coords)
                    segments.append((pts[0], pts[-1], pts))
        except Exception:
            continue
    
    if not segments:
        return []
    
    # Connect segments into closed loops
    tolerance = 1e-6
    closed_loops = []
    used = [False] * len(segments)
    
    for i in range(len(segments)):
        if used[i]:
            continue
        
        start_pt, end_pt, pts = segments[i]
        chain = list(pts)
        used[i] = True
        
        # Try to extend the chain
        max_iterations = len(segments) * 2
        for _ in range(max_iterations):
            extended = False
            current_end = chain[-1]
            
            for j in range(len(segments)):
                if used[j]:
                    continue
                
                seg_start, seg_end, seg_pts = segments[j]
                
                if math.dist(current_end, seg_start) < tolerance:
                    chain.extend(seg_pts[1:])
                    used[j] = True
                    extended = True
                    break
                elif math.dist(current_end, seg_end) < tolerance:
                    chain.extend(reversed(seg_pts[:-1]))
                    used[j] = True
                    extended = True
                    break
            
            if not extended:
                break
            
            # Check if closed loop formed
            if len(chain) >= 3 and math.dist(chain[0], chain[-1]) < tolerance:
                closed_loops.append(LineString(chain))
                break
    
    return closed_loops


def read_dxf_polygons(path: str) -> Polygon:
    """
    Import geometry from DXF file and recognize hollow/solid sections.
    Returns a composite polygon with holes properly identified.
    """
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    all_polygons = []
    
    # Process LWPOLYLINE entities
    for e in msp.query("LWPOLYLINE"):
        ls = closed_lwpolyline_to_linestring(e)
        if ls is not None:
            all_polygons.append(Polygon(ls.coords))
    
    # Process POLYLINE entities
    for e in msp.query("POLYLINE"):
        ls = polyline_to_linestring(e)
        if ls is not None:
            all_polygons.append(Polygon(ls.coords))
    
    # Process CIRCLE entities
    for e in msp.query("CIRCLE"):
        poly = circle_to_polygon(e)
        if poly is not None:
            all_polygons.append(poly)
    
    # Process LINE and ARC entities (connect into closed loops)
    line_arc_entities = list(msp.query("LINE")) + list(msp.query("ARC"))
    closed_loops = connect_entities_to_polygons(line_arc_entities)
    for ls in closed_loops:
        all_polygons.append(Polygon(ls.coords))
    
    if not all_polygons:
        raise ValueError("No closed boundaries found in DXF. Found entity types: " + 
                        ", ".join(set(e.dxftype() for e in msp)))
    
    # Recognize hollow sections by containment depth
    # Odd depth = hole, even depth = boundary
    boundaries = []
    holes = []
    
    for i, poly_i in enumerate(all_polygons):
        containment_depth = 0
        
        # Count how many LARGER polygons contain this one
        for j, poly_j in enumerate(all_polygons):
            if i != j and poly_j.area > poly_i.area:
                try:
                    test_point = poly_i.exterior.coords[0]
                    if poly_j.contains(Point(test_point)) or poly_j.touches(Point(test_point)):
                        containment_depth += 1
                except:
                    pass
        
        # Odd depth = hole (inside another polygon)
        if containment_depth % 2 == 1:
            holes.append(poly_i)
        else:
            boundaries.append(poly_i)
    
    if not boundaries:
        raise ValueError("No outer boundaries found - all polygons are nested")
    
    # Create composite geometry
    solid = unary_union(boundaries)
    if holes:
        hole_union = unary_union(holes)
        composite = solid.difference(hole_union)
    else:
        composite = solid
    
    # Handle MultiPolygon (take largest component)
    if not isinstance(composite, Polygon):
        if hasattr(composite, "geoms"):
            composite = sorted(composite.geoms, key=lambda g: g.area, reverse=True)[0]
        else:
            raise ValueError("Composite geometry is not a polygon")
    
    return composite


# ==================== MAIN COMPUTATION ====================

def compute_section_properties(poly: Polygon) -> SectionProperties:
    """
    Calculate all section properties for a given polygon.
    """
    # Validate geometry
    if not poly.is_valid:
        raise ValueError("Invalid polygon geometry")
    if poly.area <= 0:
        raise ValueError(f"Polygon has non-positive area: {poly.area}")
    
    # Basic properties
    A, perimeter, (Cx, Cy), Ixx_o, Iyy_o, Ixy_o = composite_polygon_properties(poly)
    
    # Move to centroidal axes
    Ixx_c, Iyy_c, Ixy_c = translate_moments_to_centroid(Ixx_o, Iyy_o, Ixy_o, A, Cx, Cy)
    
    # Principal moments and angle
    I_major, I_minor, angle_deg = principal_moments_and_angle(Ixx_c, Iyy_c, Ixy_c)
    angle_rad = math.radians(angle_deg)
    
    # Polar moment
    J_c = Ixx_c + Iyy_c
    
    # Extreme fiber distances along principal axes
    c_major_pos, c_major_neg, c_minor_pos, c_minor_neg = polygon_extents_along_axes(
        poly, (Cx, Cy), angle_rad)
    
    # Elastic section moduli (S = I / c)
    S_major_pos = I_major / c_major_pos if c_major_pos > 0 else float('nan')
    S_major_neg = I_major / c_major_neg if c_major_neg > 0 else float('nan')
    S_minor_pos = I_minor / c_minor_pos if c_minor_pos > 0 else float('nan')
    S_minor_neg = I_minor / c_minor_neg if c_minor_neg > 0 else float('nan')
    
    # Radii of gyration
    if I_major < 0 or I_minor < 0 or A <= 0:
        raise ValueError(f"Invalid moments: I_major={I_major}, I_minor={I_minor}, A={A}")
    
    r_gyr_major = math.sqrt(I_major / A)
    r_gyr_minor = math.sqrt(I_minor / A)
    
    # Plastic section moduli (triangulation-based approximation)
    tris = triangulate(poly)
    coords = np.array(list(poly.exterior.coords)[:-1])
    coords_rot = rotate_points(coords, angle_rad, (Cx, Cy))
    xprime = coords_rot[:, 0]
    yprime = coords_rot[:, 1]
    bounds_major = (np.min(xprime), np.max(xprime))
    bounds_minor = (np.min(yprime), np.max(yprime))
    
    Z_pl_major = plastic_section_modulus(tris, 'major', angle_rad, (Cx, Cy), bounds_major)
    Z_pl_minor = plastic_section_modulus(tris, 'minor', angle_rad, (Cx, Cy), bounds_minor)
    
    return SectionProperties(
        area=A,
        perimeter=perimeter,
        centroid=(Cx, Cy),
        Ixx_c=Ixx_c,
        Iyy_c=Iyy_c,
        Ixy_c=Ixy_c,
        principal_I_major=I_major,
        principal_I_minor=I_minor,
        principal_angle_deg=angle_deg,
        polar_J_c=J_c,
        c_major_pos=c_major_pos,
        c_major_neg=c_major_neg,
        c_minor_pos=c_minor_pos,
        c_minor_neg=c_minor_neg,
        S_major_pos=S_major_pos,
        S_major_neg=S_major_neg,
        S_minor_pos=S_minor_pos,
        S_minor_neg=S_minor_neg,
        r_gyration_major=r_gyr_major,
        r_gyration_minor=r_gyr_minor,
        Z_plastic_major=Z_pl_major,
        Z_plastic_minor=Z_pl_minor,
    )

def plot_section(poly: Polygon, sp: SectionProperties, save_path: str = None) -> str:
    """
    Visualize the section geometry with centroid and principal axes.
    Returns the path to the saved image file.
    """
    aspect_ratio = 2
    max_size = 12
    
    if aspect_ratio > 1:
        fig_width = min(max_size, aspect_ratio * 8)
        fig_height = fig_width / aspect_ratio
    else:
        fig_height = min(max_size, 8 / aspect_ratio)
        fig_width = fig_height * aspect_ratio
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Plot exterior boundary
    ext_coords = np.array(poly.exterior.coords)
    ax.plot(ext_coords[:, 0], ext_coords[:, 1], 'b-', linewidth=2, label='Boundary')
    ax.fill(ext_coords[:, 0], ext_coords[:, 1], color='lightblue', alpha=0.3)
    
    # Plot holes (if any)
    for i, ring in enumerate(poly.interiors):
        hole_coords = np.array(ring.coords)
        ax.plot(hole_coords[:, 0], hole_coords[:, 1], 'r-', linewidth=2, 
               label='Hole' if i == 0 else '')
        ax.fill(hole_coords[:, 0], hole_coords[:, 1], color='white')
    
    # Plot centroid
    cx, cy = sp.centroid
    ax.plot(cx, cy, 'ro', markersize=10, label='Centroid', zorder=5)
    
    # Plot principal axes
    angle_rad = math.radians(sp.principal_angle_deg)
    axis_length = max(sp.c_major_pos + sp.c_major_neg, sp.c_minor_pos + sp.c_minor_neg) * 0.8
    
    # Major axis
    major_x = [cx - axis_length * math.cos(angle_rad), cx + axis_length * math.cos(angle_rad)]
    major_y = [cy - axis_length * math.sin(angle_rad), cy + axis_length * math.sin(angle_rad)]
    ax.plot(major_x, major_y, 'g--', linewidth=1.5, label='Major Axis', alpha=0.7)
    
    # Minor axis (perpendicular to major)
    minor_angle = angle_rad + math.pi/2
    minor_x = [cx - axis_length * math.cos(minor_angle), cx + axis_length * math.cos(minor_angle)]
    minor_y = [cy - axis_length * math.sin(minor_angle), cy + axis_length * math.sin(minor_angle)]
    ax.plot(minor_x, minor_y, 'm--', linewidth=1.5, label='Minor Axis', alpha=0.7)
    
    ax.set_aspect('equal', 'box')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_title('Section Geometry', fontsize=12, fontweight='bold')
    
    # Save to file
    if save_path is None:
        save_path = 'section_plot.png'
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def generate_pdf_report(sp: SectionProperties, poly: Polygon, pdf_path: str = 'section_properties_report.pdf'):
    """
    Generate a PDF report of section properties with visualization.
    """
    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                           leftMargin=20*mm, rightMargin=20*mm,
                           topMargin=20*mm, bottomMargin=20*mm)
    
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    # Title
    title = Paragraph('Report for Section Properties', title_style)
    story.append(title)
    story.append(Spacer(1, 10*mm))
    
    # Generate section plot
    image_path = 'temp_section_plot.png'
    plot_section(poly, sp, image_path)
    
    # Create two-column layout
    # Left column: Properties, Right column: Image
    
    # Prepare properties data
    def format_value(v, unit=''):
        if isinstance(v, float):
            if abs(v) < 1e-6:
                return f"0.000 {unit}"
            elif abs(v) >= 1e6:
                return f"{v:.3e} {unit}"
            else:
                return f"{v:.3f} {unit}"
        return f"{v} {unit}"
    
    # Section type
    section_type = 'HOLLOW SECTION' if len(list(poly.interiors)) > 0 else 'SOLID SECTION'
    story.append(Paragraph(f'<b>Section Type:</b> {section_type}', styles['Normal']))
    if len(list(poly.interiors)) > 0:
        story.append(Paragraph(f'<b>Number of Holes:</b> {len(list(poly.interiors))}', styles['Normal']))
    story.append(Spacer(1, 10*mm))
    
    # PAGE 1: Full page section image
    from reportlab.platypus import PageBreak
    
    # Add large section image
    section_img = Image(image_path, width=160*mm, height=160*mm)
    story.append(section_img)
    
    # Page break to move to next page
    story.append(PageBreak())
    
    # PAGE 2: Section properties table
    # Add title again on second page
    title_page2 = Paragraph('Section Properties', title_style)
    story.append(title_page2)
    story.append(Spacer(1, 10*mm))
    
    # Properties table data
    properties_data = [
        ['Property', 'Value', 'Unit'],
        ['', '', ''],  # Separator
        ['GEOMETRIC PROPERTIES', '', ''],
        ['Area', f"{sp.area:.0f}", Paragraph('mm<super>2</super>', styles['Normal'])],
        ['Perimeter', f"{sp.perimeter:.1f}", Paragraph('mm', styles['Normal'])],
        ['Centroid X', f"{sp.centroid[0]:.1f}", Paragraph('mm', styles['Normal'])],
        ['Centroid Y', f"{sp.centroid[1]:.1f}", Paragraph('mm', styles['Normal'])],
        ['', '', ''],
        ['MOMENTS OF INERTIA', '', ''],
        ['Ixx (about centroid)', f"{sp.Ixx_c:.2e}", Paragraph('mm<super>4</super>', styles['Normal'])],
        ['Iyy (about centroid)', f"{sp.Iyy_c:.2e}", Paragraph('mm<super>4</super>', styles['Normal'])],
        ['Ixy (about centroid)', f"{sp.Ixy_c:.2e}", Paragraph('mm<super>4</super>', styles['Normal'])],
        ['Polar Moment J', f"{sp.polar_J_c:.2e}", Paragraph('mm<super>4</super>', styles['Normal'])],
        ['', '', ''],
        ['PRINCIPAL MOMENTS', '', ''],
        ['I major', f"{sp.principal_I_major:.2e}", Paragraph('mm<super>4</super>', styles['Normal'])],
        ['I minor', f"{sp.principal_I_minor:.2e}", Paragraph('mm<super>4</super>', styles['Normal'])],
        ['Principal angle', f"{sp.principal_angle_deg:.1f}", Paragraph('°', styles['Normal'])],
        ['', '', ''],
        ['DISTANCES TO EXTREME FIBERS', '', ''],
        ['c major (+)', f"{sp.c_major_pos:.1f}", Paragraph('mm', styles['Normal'])],
        ['c major (-)', f"{sp.c_major_neg:.1f}", Paragraph('mm', styles['Normal'])],
        ['c minor (+)', f"{sp.c_minor_pos:.1f}", Paragraph('mm', styles['Normal'])],
        ['c minor (-)', f"{sp.c_minor_neg:.1f}", Paragraph('mm', styles['Normal'])],
        ['', '', ''],
        ['ELASTIC SECTION MODULI', '', ''],
        ['S major (+)', f"{sp.S_major_pos:.2e}", Paragraph('mm<super>3</super>', styles['Normal'])],
        ['S major (-)', f"{sp.S_major_neg:.2e}", Paragraph('mm<super>3</super>', styles['Normal'])],
        ['S minor (+)', f"{sp.S_minor_pos:.2e}", Paragraph('mm<super>3</super>', styles['Normal'])],
        ['S minor (-)', f"{sp.S_minor_neg:.2e}", Paragraph('mm<super>3</super>', styles['Normal'])],
        ['', '', ''],
        ['RADII OF GYRATION', '', ''],
        ['r major', f"{sp.r_gyration_major:.1f}", Paragraph('mm', styles['Normal'])],
        ['r minor', f"{sp.r_gyration_minor:.1f}", Paragraph('mm', styles['Normal'])],
        ['', '', ''],
        ['PLASTIC SECTION MODULI', '', ''],
        ['Z major', f"{sp.Z_plastic_major:.2e}", Paragraph('mm<super>3</super>', styles['Normal'])],
        ['Z minor', f"{sp.Z_plastic_minor:.2e}", Paragraph('mm<super>3</super>', styles['Normal'])],
    ]
    
    # Create properties table (full width on second page)
    prop_table = Table(properties_data, colWidths=[110*mm, 40*mm, 20*mm])
    prop_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('FONTNAME', (0, 2), (0, 2), 'Helvetica-Bold'),
        ('FONTNAME', (0, 8), (0, 8), 'Helvetica-Bold'),
        ('FONTNAME', (0, 14), (0, 14), 'Helvetica-Bold'),
        ('FONTNAME', (0, 19), (0, 19), 'Helvetica-Bold'),
        ('FONTNAME', (0, 25), (0, 25), 'Helvetica-Bold'),
        ('FONTNAME', (0, 31), (0, 31), 'Helvetica-Bold'),
        ('FONTNAME', (0, 35), (0, 35), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')]),
        ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#e8f4f8')),
        ('BACKGROUND', (0, 8), (-1, 8), colors.HexColor('#e8f4f8')),
        ('BACKGROUND', (0, 14), (-1, 14), colors.HexColor('#e8f4f8')),
        ('BACKGROUND', (0, 19), (-1, 19), colors.HexColor('#e8f4f8')),
        ('BACKGROUND', (0, 25), (-1, 25), colors.HexColor('#e8f4f8')),
        ('BACKGROUND', (0, 31), (-1, 31), colors.HexColor('#e8f4f8')),
        ('BACKGROUND', (0, 35), (-1, 35), colors.HexColor('#e8f4f8')),
    ]))
    
    story.append(prop_table)
    
    # Build PDF
    doc.build(story)
    
    # Clean up temporary image
    if os.path.exists(image_path):
        os.remove(image_path)
    
    print(f"\nPDF report generated: {pdf_path}")


def print_section_properties(sp: SectionProperties):
    """Display calculated section properties in formatted output."""
    def f(v): return f"{v:.6f}"
    
    print("\n" + "="*60)
    print("SECTION PROPERTIES")
    print("="*60)
    print(f"\nGEOMETRIC PROPERTIES:")
    print(f"  Area:                    {f(sp.area)}")
    print(f"  Perimeter:               {f(sp.perimeter)}")
    print(f"  Centroid (x, y):         ({f(sp.centroid[0])}, {f(sp.centroid[1])})")
    
    print(f"\nMOMENTS OF INERTIA (about centroid):")
    print(f"  Ixx:                     {f(sp.Ixx_c)}")
    print(f"  Iyy:                     {f(sp.Iyy_c)}")
    print(f"  Ixy:                     {f(sp.Ixy_c)}")
    print(f"  Polar J:                 {f(sp.polar_J_c)}")
    
    print(f"\nPRINCIPAL MOMENTS:")
    print(f"  I_major:                 {f(sp.principal_I_major)}")
    print(f"  I_minor:                 {f(sp.principal_I_minor)}")
    print(f"  Principal angle:         {f(sp.principal_angle_deg)}°")
    
    print(f"\nDISTANCES TO EXTREME FIBERS:")
    print(f"  c_major (+/-):           {f(sp.c_major_pos)} / {f(sp.c_major_neg)}")
    print(f"  c_minor (+/-):           {f(sp.c_minor_pos)} / {f(sp.c_minor_neg)}")
    
    print(f"\nELASTIC SECTION MODULI:")
    print(f"  S_major (+/-):           {f(sp.S_major_pos)} / {f(sp.S_major_neg)}")
    print(f"  S_minor (+/-):           {f(sp.S_minor_pos)} / {f(sp.S_minor_neg)}")
    
    print(f"\nRADII OF GYRATION:")
    print(f"  r_major:                 {f(sp.r_gyration_major)}")
    print(f"  r_minor:                 {f(sp.r_gyration_minor)}")
    
    print(f"\nPLASTIC SECTION MODULI:")
    print(f"  Z_major:                 {f(sp.Z_plastic_major)}")
    print(f"  Z_minor:                 {f(sp.Z_plastic_minor)}")
    print("="*60 + "\n")


def main():
    """Main entry point for section properties calculator."""
    parser = argparse.ArgumentParser(
        description="Section Properties Calculator - Import DXF and calculate comprehensive section properties",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python section_properties.py drawing.dxf
  python section_properties.py hollow_section.dxf
        """
    )
    parser.add_argument("path", type=str, nargs='?', help="Path to DXF file")
    args = parser.parse_args()

    if not args.path:
        args.path = input("Enter path to DXF file: ").strip()

    try:
        print(f"\nImporting geometry from: {args.path}")
        poly = read_dxf_polygons(args.path)
        
        print(f"Recognized section type: {'HOLLOW' if len(list(poly.interiors)) > 0 else 'SOLID'}")
        if len(list(poly.interiors)) > 0:
            print(f"Number of holes: {len(list(poly.interiors))}")
        
        sp = compute_section_properties(poly)
        print_section_properties(sp)
        
        # Ask if user wants to generate PDF report
        response = input("\nDo you want to generate a PDF report? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            # Generate default filename based on input file
            base_name = os.path.splitext(os.path.basename(args.path))[0]
            pdf_filename = f"{base_name}_section_properties_report.pdf"
            
            generate_pdf_report(sp, poly, pdf_filename)
            print(f"PDF report saved as: {pdf_filename}")
        
    except FileNotFoundError:
        print(f"Error: File '{args.path}' not found")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
