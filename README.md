# 🌌 StarWeb-CARA: Conjunction Assessment and Collision Risk Analysis

![StarWeb-CARA Interface](https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/world.topo.bathy.200412.3x5400x2700.jpg)
> *A comprehensive Streamlit-based web application for analyzing conjunction events and collision risks in Low Earth Orbit (LEO).*

This system implements industry-standard algorithms for satellite collision probability assessment, including multiple Pc models, fragmentation analysis, and real-time cinematic 3D visualizations.

---

## 🎓 Academic Context & Credits

### Why This Project Was Developed
This system was developed as a **Graduation Project** for the **Space Sciences and Technologies** department. With the exponential rise of low Earth orbit (LEO) satellite deployments, monitoring and mitigating orbital debris risks has become a critical challenge for modern space safety. 

The primary objective of this project is to implement, evaluate, and visualize industry-standard conjunction assessment risk analysis (CARA) methodologies specifically focusing on the operational collision risks posed by modern megaconstellations such as Starlink and OneWeb.

### 👥 Project Team
* **Author:** **ALTAY ÇAVUŞ**
    * **Role:** Lead Developer & Researcher 
    * **Department:** Space Sciences and Technologies
    * **Contact:** *altaycavuss@gmail.com* *
    * **Academic Advisor:** **Assoc. Prof. BURCU ÖZKARDEŞ**
    * **Role:** Project Supervisor / Consultant 
    * **Affiliation:** Department of Space Sciences and Technologies

---

## 🚀 Key Features & Innovations

### 1. Core Analysis Capabilities
- **Live TLE Integration**: Connects directly with the *Space-Track GP* database and *CelesTrak* (fallback) for real-time orbital data.
- **Megaconstellation Support**: Instantly loads fleets like STARLINK, ONEWEB, ISS, KUIPER, IRIDIUM-NEXT, and PLANET.
- **Custom Satellite Entry**: Inject your own 3-line TLE data to test your custom satellite against existing space infrastructure.
- **Apsis Filter (ESA/NASA Standard)**: Dramatically reduces O(N²) computational load by pre-filtering satellite pairs with non-overlapping altitude bands.

### 2. Advanced Collision Probability (Pc) Models
- **Chan (1997) Isotropic Model**: Fast closed-form approximation assuming spherical position uncertainty.
- **Foster & Estes (1992) 2D-Pc**: The ultimate industry standard. Projects uncertainties onto a 2D encounter plane and integrates the combined Gaussian probability density over the Hard-Body Radius (HBR).
- **Max-Pc Analysis**: Iterative covariance variation to find the mathematical maximum collision probability for a given geometry.
- **Mahalanobis Distance Test**: Evaluates the validity of 2D-Pc assumptions (Md < 1.5 → 3D-Pc required).

### 3. Risk Assessment & Physics
- **NASA STD-8719.14 Risk Classification**: Four-tier risk classification (CRITICAL, HIGH, MEDIUM, LOW).
- **Probability Dilution Detection**: Alerts users to "false confidence" scenarios where wide position uncertainty mathematically drops the Pc.
- **Fragmentation Probability (Pf)**: Calculates Specific Kinetic Energy ($E_c = \frac{1}{2} m_b v_{rel}^2 / m_a$) to determine if a collision would cause Catastrophic Fragmentation (Kessler Syndrome).

### 4. Cinematic 3D Visualization & Simulators
- **Live 3D Encounter Simulator**: Watch two satellites approach each other in real-time. Features a "Jump to TCA" button, precise 5-minute steps, and a dynamic HUD (Heads-Up Display) popping up at the moment of closest approach.
- **Cyberpunk Orbital Web**: Real-time rendering of the Earth using NASA's *Blue Marble* texture with atmospheric glow, equator/meridian lines, and futuristic node-based orbit trails.
- **Data Export & Radars**: 3x3 metric grids for Keplerian elements, Altitude vs. Inclination scatter plots, and one-click Excel (CSV) exports for both conjunction reports and orbital profiles.

---

## 📋 Methodology Deep-Dive

**1. Orbit Propagation (SGP4/SDP4)**
The NORAD standard SGP4 model (via the `Skyfield` Python library) calculates position vectors by accounting for gravity harmonics, atmospheric drag, and third-body effects. 

**2. TCA Detection (5-Minute Coarse Scan)**
For pairs passing the Apsis filter, Euclidean distances are calculated iteratively with strict **5-minute fixed time steps**. The global minimum in this array determines the exact **Time of Closest Approach (TCA)**.

**3. Specific Kinetic Energy & Debris**
If a collision occurs, its severity matters. The system calculates $E_c$ (J/g). If $E_c \geq 40$ J/g, it's flagged as a **Catastrophic Fragmentation**, meaning the satellite completely shatters, significantly contributing to orbital debris.

---

## 💻 Installation & Usage

### Prerequisites
Before you begin, ensure you have met the following requirements:
- **Python 3.9 or higher** installed on your system.
- **Git** installed.
- A free account at [Space-Track.org](https://www.space-track.org) (required for pulling live TLE orbital data).

### Step-by-Step Installation

**1. Clone the repository** Open your terminal or command prompt and clone this repository to your local machine:
```bash
git clone [https://github.com/altonythecoder/StarWeb-CARA.git](https://github.com/altonythecoder/StarWeb-CARA.git)
cd StarWeb-CARA

2. Create a Virtual Environment (Recommended) It is highly recommended to use a virtual environment to isolate project dependencies and avoid conflicts.

For Windows:
python -m venv venv
venv\Scripts\activate

For macOS and Linux:
python3 -m venv venv
source venv/bin/activate

Install Dependencies Install the required Python packages using the provided requirements.txt file. This will install core libraries such as streamlit, skyfield, plotly, scipy, and spacetrack:
pip install -r requirements.txt

Launch the Application Start the Streamlit web server. This will automatically open the application in your default web browser:
streamlit run starweb-cara.py

🚀 Quick Start Guide (In-App)
Once the app launches, locate the Control Panel on the left sidebar.
Enter your Space-Track.org email and password in the authentication section.
Select a target constellation (e.g., STARLINK, ONEWEB, ISS, KUIPER).
Click DOWNLOAD LIVE TLE DATA to fetch the real-time orbital elements.
Adjust your analysis parameters (Time Window, Position Uncertainty σ, Hard-Body Radius, Satellite Masses).
Explore the Live Simulation, Conjunction Analysis, and 3D Orbit tabs to analyze collision risks!
