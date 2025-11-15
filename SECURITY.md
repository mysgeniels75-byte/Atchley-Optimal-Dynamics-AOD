# Security Policy

## Project Status

The Atchley Optimal Dynamics (AOD) Theory is an **open-source research project** focused on fundamental theoretical neuroscience and computational optimization.

**This is scientific research code, not production software.**

## Supported Versions

| Version | Status | Support |
|---------|--------|---------|
| 2.0.x (Phase 2) | ✅ Current | Full support |
| 1.0.x (Phase 1) | ⚠️ Legacy | Security fixes only |
| < 1.0 | ❌ Unsupported | No support |

## Security Considerations

### Research Code

This codebase is intended for:
- Scientific research and validation
- Educational purposes
- Academic experimentation
- Proof-of-concept demonstrations

**NOT intended for:**
- Production systems
- Safety-critical applications
- Medical devices
- Financial systems
- Any deployment where failure could cause harm

### Data Privacy

- **No Personal Data Collection**: This project does not collect, store, or transmit personal information
- **Simulation Data Only**: All data is synthetically generated for research purposes
- **Neural Data**: If real neural recordings are used (future), proper IRB approval and consent required

### Computational Security

**Safe Practices:**
- All code runs in local environment
- No external API calls (except optional cloud computing)
- Deterministic simulations (reproducible results)
- No code execution from untrusted sources

**Potential Risks:**
- Large simulations may consume significant CPU/GPU resources
- Memory usage can grow for large populations (monitor RAM)
- Long-running evolutionary algorithms may run for hours/days

### Dependencies

Current dependencies (see `requirements.txt`):
- `numpy>=1.20.0` - Numerical computation
- `matplotlib>=3.3.0` - Visualization
- `scipy>=1.7.0` - Scientific computing

**Security Updates:**
- Dependencies are regularly updated
- Known vulnerabilities are monitored via GitHub Dependabot
- Users should `pip install --upgrade` regularly

## Reporting a Vulnerability

### What to Report

Please report security issues if you discover:

1. **Code Vulnerabilities**
   - Arbitrary code execution
   - Unsafe deserialization
   - Path traversal
   - Injection vulnerabilities

2. **Dependency Issues**
   - Known CVEs in dependencies
   - Outdated libraries with security patches available

3. **Data Leakage**
   - Unintended data exposure
   - Privacy violations (if neural data added later)

### What NOT to Report

The following are **not security issues** for this research project:

- Performance issues (slow code)
- Scientific correctness (see Issues for bugs)
- Missing features
- Documentation errors
- Theoretical disagreements

### How to Report

**For Security Issues:**

1. **DO NOT** open a public GitHub issue
2. **Email**: (Add security contact email when available)
   - Subject: "AOD Security Issue"
   - Include: Description, steps to reproduce, potential impact
3. **GitHub Security Advisory**: Use the "Security" tab → "Report a vulnerability"

**Response Timeline:**
- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 7 days
- **Fix & Disclosure**: Coordinated with reporter

### Disclosure Policy

- **Responsible Disclosure**: 90-day embargo before public disclosure
- **Credit**: Security researchers will be credited (unless they prefer anonymity)
- **CVE Assignment**: For critical vulnerabilities (if applicable)

## Security Best Practices for Users

### Installation

```bash
# Use virtual environment to isolate dependencies
python -m venv aod_env
source aod_env/bin/activate  # On Windows: aod_env\Scripts\activate

# Install from requirements
pip install -r requirements.txt

# Verify checksums (when available)
# sha256sum -c checksums.txt
```

### Running Simulations

```bash
# Limit computational resources if needed
export OMP_NUM_THREADS=4  # Limit parallel threads
ulimit -m 8000000         # Limit memory (8GB)

# Run with resource monitoring
python aod_mvp_demo.py
```

### Code Review

If modifying the code:
- Review all external inputs
- Validate file paths (prevent directory traversal)
- Sanitize any user-provided data
- Use type hints and assertions
- Run tests before deployment

## Research Ethics

### Scientific Integrity

- **No Fabrication**: All results are from actual simulations
- **Reproducibility**: Random seeds provided for deterministic results
- **Transparency**: Full code and data available for verification

### AI Ethics

This project adheres to principles of:
- **Beneficial AI**: Focus on efficiency, not surveillance
- **No Dual Use**: Reject military/weaponization applications
- **Privacy First**: No personal data collection
- **Open Science**: All research publicly accessible

### Data Handling (Future)

If neural data is incorporated:
- **IRB Approval Required**: All human/animal studies must have ethical approval
- **Informed Consent**: Participants must consent to data use
- **Anonymization**: Remove all personally identifiable information
- **Secure Storage**: Encrypted storage, access controls
- **Data Use Agreement**: Respect original study's restrictions

## Threat Model

### In Scope

- Vulnerabilities in our code
- Dependency vulnerabilities
- Build/deployment security
- Data handling (if applicable)

### Out of Scope

- Social engineering
- Physical attacks
- DDoS against GitHub
- Vulnerabilities in user's environment
- Theoretical attacks on the AOD algorithm itself

## Compliance

### Licensing

- **MIT License**: Open-source, permissive
- **No Warranties**: Provided "as-is"
- **Attribution Required**: Credit original authors

### Export Control

This project contains:
- Publicly available research code
- No encryption (except dependencies)
- No restricted technologies

**Not subject to** U.S. export control (EAR/ITAR).

## Security Roadmap

### Current (Phase 2)

- ✅ Secure coding practices
- ✅ Dependency monitoring
- ✅ Open-source transparency

### Planned (Phase 3)

- 🔄 Automated security scanning (Snyk, CodeQL)
- 🔄 Signed releases (GPG signatures)
- 🔄 Container security (Docker image scanning)
- 🔄 Supply chain security (SBOM generation)

### Future

- Hardware security (if memristor prototypes built)
- Neuromorphic chip security
- Federated learning security (if multi-party computation added)

## Contact

**Project Maintainer**: Devin Earl Atchley

**Security Contact**: (Add email when available)

**GitHub**: [@mysgeniels75-byte](https://github.com/mysgeniels75-byte)

**For non-security issues**: Use [GitHub Issues](https://github.com/mysgeniels75-byte/Atchley-Optimal-Dynamics-AOD/issues)

---

## Acknowledgments

We thank the security research community for responsible disclosure practices and helping keep open-source research secure.

---

*Last Updated: November 2024*
*Security Policy Version: 1.0*
