# LLM-Assisted Longitudinal Adverse Event Tracking in Clinical Trials

AI-assisted framework for standardized, longitudinal CTCAE-based adverse event (AE) extraction, normalization, grading, and patient-level timeline consolidation in oncology clinical trials.

---

## Overview

Adverse event (AE) documentation is central to safety monitoring in oncology clinical trials. However, AE identification, CTCAE mapping, grading, and attribution remain largely manual, labor-intensive, and inconsistent across visits.

We present an end-to-end **LLM-assisted framework** that:

- Extracts AEs from clinical notes and laboratory records  
- Normalizes events to **CTCAE v5.0**  
- Assigns severity grade and attribution  
- Flags immune-related and serious AEs  
- Maintains **patient-level longitudinal AE trajectories**

The system integrates LLM reasoning with ontology-guided semantic retrieval and deterministic grading logic to support real-world safety reporting workflows.

---

## Clinical Context

### CTCAE-Based Safety Reporting

Adverse events in oncology trials are standardized using:

- **National Cancer Institute (NCI)**
- **Common Terminology Criteria for Adverse Events (CTCAE v5.0)**

CTCAE defines:

- Preferred AE terms  
- Severity grades (1–5)  
- Standardized terminology for regulatory reporting  

This framework explicitly aligns extraction and normalization to CTCAE v5.0.

---

## Framework Architecture

### End-to-End Pipeline
