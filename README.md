## 🎓 Academic Context & Credits

### Why This Project Was Developed
This system was developed as a **Graduation Project** for the **Space Sciences and Technologies** department. With the exponential rise of low Earth orbit (LEO) satellite deployments, monitoring and mitigating orbital debris risks has become a critical challenge for modern space safety. 

The primary objective of this project is to implement, evaluate, and visualize industry-standard conjunction assessment risk analysis (CARA) methodologies specifically focusing on the operational collision risks posed by modern megaconstellations such as Starlink and OneWeb.

### 👥 Project Team

* **Author:** **ALTAY ÇAVUŞ**
    * **Role:** Lead Developer & Researcher 
    * **Department:** Space Sciences and Technologies
    * **Contact:** *altaycavuss@gmail.com* 

* **Academic Advisor:** **Assoc. Prof. BURCU ÖZKARDEŞ**
    * **Role:** Project Supervisor / Consultant 
    * **Affiliation:** Department of Space Sciences and Technologies

# StarWeb-CARA: Conjunction Assessment and Collision Risk Analysis for Starlink and OneWeb Megaconstellations

A comprehensive Streamlit-based web application for analyzing conjunction events and collision risks in Low Earth Orbit (LEO). This system implements industry-standard algorithms for satellite collision probability assessment, including multiple Pc models, fragmentation analysis, and real-time 3D visualization.

## 🚀 Features

### Core Analysis Capabilities
- **Live TLE Data Download**: Direct integration with Space-Track GP database for real-time satellite orbital data
- **Multiple Constellation Support**: Pre-configured support for STARLINK, ISS, and ONEWEB constellations
- **Custom Satellite Analysis**: Enter your own TLE data to analyze conjunction risks with existing satellite fleets
- **Apsis Filter**: ESA/NASA standard pre-filtering algorithm that dramatically reduces O(N²) computational load by eliminating satellite pairs with non-overlapping altitude bands

### Collision Probability Models
- **Chan (1997) Isotropic Model**: Fast closed-form approximation assuming spherical position uncertainty distribution
- **Foster & Estes (1992) 2D-Pc**: Industry standard model used since NASA Space Shuttle era, projecting collision integration onto the encounter plane
- **Max-Pc Analysis**: Iterative covariance variation to find mathematical maximum collision probability for given geometry
- **Mahalanobis Distance Test**: CARA methodology validity test for 2D-Pc assumptions (Md < 1.5 → 3D-Pc required)

### Risk Assessment
- **NASA STD-8719.14 Risk Classification**: Four-tier risk classification system (CRITICAL, HIGH, MEDIUM, LOW)
- **Probability Dilution Detection**: Identifies false confidence scenarios where wide covariance masks true collision risk
- **Fragmentation Probability Analysis**: Kinetic energy-based fragmentation risk assessment per NASA operational guidelines
- **Specific Kinetic Energy Calculation**: Ec = ½ · m_b · v_rel² / m_a (J/g) with configurable satellite masses (10-5000 kg)

### Visualization
- **3D Orbit Visualization**: Interactive 3D Plotly visualization of satellite orbits with Earth texture
- **Ground Track Maps**: Real-time ground track visualization with 95-minute orbital paths
- **Live 3D Animation**: Real-time animation of satellite encounters with Play/Stop/Speed controls and Jump to TCA functionality
- **Distance Profile Plots**: Time-series distance profiles showing TCA (Time of Closest Approach)
- **Risk Gauge Visualizations**: Interactive gauge displays for collision probability levels
- **Orbital Elements Radar**: Altitude/Inclination distribution visualization with eccentricity-based sizing

### User Interface
- **Mission Control Dark Theme**: Professional dark-themed UI with Space Mono and Barlow Condensed fonts
- **Seven Interactive Tabs**: Dashboard, Conjunction Analysis, Your Satellite, Live Simulation, 3D Orbit & Ground Track, Orbital Elements, Methodology
- **CSV Export**: Download conjunction analysis reports as CSV files
- **Responsive Design**: Wide layout optimized for desktop viewing

## 📋 Methodology

### 1. Orbit Propagation — SGP4/SDP4 (Skyfield)
The NORAD standard Simplified General Perturbations-4 (SGP4) model converts TLE (Two-Line Element) data into position vectors. SGP4 approximately accounts for gravity harmonics, atmospheric drag, and Sun/Moon third-body effects with a centered force model. SGP4 is used for low-orbit (<2000 km) objects; SDP4 automatically engages for high-orbit objects.

### 2. Apsis Filter — Section 2.1 (ESA/NASA Standard)
Before analysis begins, all satellite pairs pass through the Apsis (Apogee-Perigee) Filter. If the first object's perigee altitude q₁ is higher than the second object's apogee altitude Q₂, these two orbits can never intersect in space. Mathematical condition: `max(q₁, q₂) > min(Q₁, Q₂) + D_th`. This filter dramatically reduces O(N²) computational load.

### 3. TCA Detection — 5-Minute Step Coarse Scan
For pairs passing the apsis filter, Euclidean distance is calculated throughout the analysis window with 5-minute fixed time steps. The moment of minimum distance is identified as TCA (Time of Closest Approach). Local minimization with Brent's method can be applied for more precise TCA.

### 4. Collision Probability — Two Models
- **Chan (1997) Isotropic Model**: Simplified model assuming position uncertainty is equally (spherically) distributed in all directions. Formula: normal CDF-based closed-form approximation.
- **Foster & Estes (1992) 2D-Pc**: Industry standard used since NASA Space Shuttle era. Collision integration is reduced to two dimensions by projecting onto the encounter plane. Combined covariance matrix (Σ = Cₐ + C_b) is created and the 2D integral of Gaussian distribution over HBR circle is calculated.

### 5. Mahalanobis Distance Test — Section 3.2 (CARA Methodology)
2D-Pc's "short-duration encounter" and "linear motion" assumptions break down when objects approach at low relative velocities. According to CARA methodology, Mahalanobis distance (Md = miss / σ) tests this validity:
- Md < 0.5 → 2D-Pc INVALID — 3D-Pc / Monte Carlo required
- Md < 1.5 → 2D-Pc BORDERLINE — 3D-Pc recommended
- Md ≥ 1.5 → 2D-Pc VALID

### 6. Probability Dilution — Section 4
With large position uncertainty (wide covariance), the Gaussian distribution spreads so much in space that the density falling within the HBR circle approaches zero — Pc mathematically decreases. This "false confidence" problem can make a genuinely dangerous close approach appear safe. Solution tools: WSPRT (Wald Sequential Probability Ratio Test) and Max-Pc Analysis.

### 7. Risk Classification — NASA STD-8719.14
- Pc > 1×10⁻³ → CRITICAL — Collision Avoidance Maneuver (CAM) mandatory
- Pc > 1×10⁻⁴ → HIGH — CAM evaluation required
- Pc > 1×10⁻⁵ → MEDIUM — Increased tracking frequency
- Pc ≤ 1×10⁻⁵ → LOW — Routine tracking sufficient

### 8. Collision Consequence — Fragmentation Probability Pf
Not only collision probability, but also the magnitude of potential disaster should be included in risk calculation. Specific Kinetic Energy: Ec = ½ · m_b · v_rel² / m_a (J/g)
- Ec ≥ 40 J/g → Catastrophic fragmentation — Kessler Syndrome contribution
- Ec ≥ 10 J/g → Severe damage and significant debris cloud
- Ec ≥ 1 J/g → Partial damage

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Space-Track account (free registration at space-track.org)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/altonythecoder/StarWeb-CARA
cd StarWeb-CARA
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install streamlit pandas numpy plotly skyfield spacetrack scipy pillow requests
```

4. Run the application:
```bash
streamlit run starweb-cara.py
```

The application will open in your default web browser at `http://localhost:8501`.

## 📖 Usage

### Getting Started

1. **Space-Track Authentication**: Enter your Space-Track email and password in the left sidebar control panel
2. **Select Constellation**: Choose between STARLINK, ISS, or ONEWEB constellations
3. **Download Data**: Click "DOWNLOAD LIVE TLE DATA" to fetch current orbital data
4. **Configure Parameters**: Adjust analysis parameters in the sidebar:
   - Analysis window (hours): 1-48 hours (default: 24)
   - Position uncertainty σ (km): 0.05-5.0 km (default: 0.5)
   - Maximum satellite count: 5-30 satellites (default: 15)
   - Hard-Body Radius HBR (km): 0.005-0.100 km (default: 0.020)
   - Object A Mass (kg): 10-5000 kg (default: 250)
   - Object B Mass (kg): 10-5000 kg (default: 250)

### Using the Tabs

**Dashboard**: Overview of conjunction events with risk metrics, apsis filter statistics, and downloadable CSV reports

**Conjunction Analysis**: Detailed review of individual conjunction events with distance profiles, collision probability model comparisons, Mahalanobis test results, and fragmentation analysis

**Your Satellite**: Analyze your own satellite's collision risk with existing fleets. Enter your TLE data in the sidebar under "2 — ENTER YOUR SATELLITE (TLE)"

**Live Simulation**: Watch real-time 3D animation of satellite encounters with Play/Stop/Speed controls and Jump to TCA functionality

**3D Orbit & Ground Track**: Visualize satellite orbits in 3D space and ground track maps

**Orbital Elements**: View Kepler orbital elements table and altitude/inclination distribution

**Methodology**: Detailed theoretical background and algorithm explanations

### Custom Satellite Analysis

To analyze your own satellite:

1. Navigate to the sidebar section "2 — ENTER YOUR SATELLITE (TLE)"
2. Enter your TLE data in the text area (3-line format: name + line1 + line2)
3. Click "LOAD MANUAL TLE"
4. Go to the "YOUR SATELLITE" tab to view conjunction analysis results

## 📚 Dependencies

- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **plotly**: Interactive visualization library
- **skyfield**: Astronomical calculations and SGP4/SDP4 orbit propagation
- **spacetrack**: Space-Track API client
- **scipy**: Scientific computing (statistics, integration)
- **pillow**: Image processing
- **requests**: HTTP library

## 📖 References

- Foster, J.L. & Estes, H.S. (1992). A parametric analysis of orbital debris collision probability and maneuver rate for space vehicles. NASA Technical Memorandum.
- Chan, F.K. (1997). Spacecraft Collision Probability. The Aerospace Press.
- Hoots, F.R. & Roehrich, R.L. (1980). Models for Propagation of NORAD Element Sets. Spacetrack Report No. 3.
- NASA (2023). Spacecraft Conjunction Assessment and Collision Avoidance Best Practices Handbook. CARA Handbook Rev. 1.
- NASA (2011). Process for Limiting Orbital Debris. NASA-STD-8719.14A.
- Alfriend, K.T. & Akella, M.R. (2000). Probability of Collision Between Space Objects. J. Guidance, Control, and Dynamics, 23(5), 769–772.
- ESA (2011). Efficient All vs. All Collision Risk Analyses — Smart Sieve Algorithm. ISSFD Proceedings.
- Vallado, D.A. (2013). Fundamentals of Astrodynamics and Applications. 4th ed. Microcosm Press.
- Hall, D.T. et al. (2023). A Multistep Probability of Collision Computational Algorithm. NASA NTRS.

## 🔧 Technical Details

- **Orbit Propagation**: SGP4/SDP4 via Skyfield library
- **Data Source**: Space-Track GP endpoint (18th Space Defense Squadron, US Space Force)
- **Coordinate System**: ECI (Earth-Centered Inertial) for calculations, ECEF (Earth-Centered Earth-Fixed) for ground tracks
- **Time System**: UTC (Coordinated Universal Time)
- **Performance Note**: Pure Python/Skyfield produces ~1M steps/sec. For large-scale operational simulations, Rust/Zig-based libraries like Astrora (4.8–15M steps/sec with SIMD) or SatKit (~3.4M steps/sec) are recommended.

## 📄 License

This project is provided for educational and research purposes. Please refer to individual library licenses for dependency licensing information.

## 🤝 Contributing

This is an academic project. For questions, suggestions, or collaboration opportunities, please open an issue or contact the maintainers.

## ⚠️ Disclaimer

This system is provided for educational and research purposes only. Collision probability calculations are approximations and should not be used for operational collision avoidance decisions without proper validation and expert review. Always consult official conjunction assessment services for operational satellite operations.

## 📧 Contact

For questions or support regarding this project, please contact the maintainers through the repository's issue tracker.

---

**Space Sciences and Technologies Graduation Project**  
*Developed with ❤️ for the space community*
