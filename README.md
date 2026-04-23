# Section Properties Calculator with CAD

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Engineering](https://img.shields.io/badge/Field-Structural%20Engineering-blue.svg)]()

A high-performance Python tool designed to calculate structural section properties (Area, Moment of Inertia, Centroid) directly from CAD files (.dxf) or image-based cross-sections using computer vision.

---

## 📖 Overview
This repository provides an automated pipeline to bridge the gap between raw geometry and structural analysis. Whether you have a legacy hand-drawn sketch or a complex built-up DXF profile, this tool extracts the boundary and computes the technical data required for engineering design.

## 🧮 Methodology
The calculator utilizes Green's Theorem to evaluate area integrals along the boundary of the detected shapes:
    <p align="center">
        <font size="100">$$A = \iint_D dA = \oint_{\partial D} x \, dy$$ </font>
    </p>
# $$I_x = \iint_D y^2 \, dA = \oint_{\partial D} \frac{y^3}{3} \, dx$$

## 🛠 1. Installation

### Clone the Repository
    git clone [https://github.com/kang851216/section-properties-calculator-with-CAD.git](https://github.com/kang851216/section-properties-calculator-with-CAD.git)
    cd section-properties-calculator-with-CAD


## 📦 2. Install Dependencies
# This project relies on OpenCV for image processing and ezdxf for CAD parsing. Install all requirements using the following command:
    pip install -r requirements.txt

## 🚀 3. Quick Start
# Process a CAD File (.dxf)
# To calculate properties for a specific CAD drawing, run:
    python main.py --input ./examples/beam_section.dxf

# Process an Image (.png/.jpg)
# To extract a section from an image via edge detection, specify the file and a scale factor:
    python main.py --image ./examples/test_section.png --scale 1.0
