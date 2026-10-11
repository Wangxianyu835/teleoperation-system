"""URDF metadata reading; no simulation or algorithm execution."""

def parse_joint_limits(urdf_path):
    """从 URDF 读关节限位 {关节名: (lower, upper)}"""
    import xml.etree.ElementTree as ET
    root = ET.parse(urdf_path).getroot()
    out = {}
    for j in root.findall('joint'):
        if j.get('type') == 'fixed':
            continue
        lim = j.find('limit')
        if lim is None:
            continue
        try:
            out[j.get('name')] = (float(lim.get('lower', -3.15)),
                                  float(lim.get('upper', 3.15)))
        except (TypeError, ValueError):
            pass
    return out

