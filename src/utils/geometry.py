import numpy as np

def calculate_angle(a, b, c):
    """
    Calculates the angle between three points (a, b, c) where b is the vertex.
    Points should be in (x, y) or (x, y, z) format.
    
    Args:
        a: First point [x, y]
        b: Vertex point [x, y]
        c: Third point [x, y]
        
    Returns:
        float: Angle in degrees in range [0, 180].
    """
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    # Calculate vectors
    ba = a - b
    bc = c - b

    # Use dot product and magnitudes to find the angle
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)

def get_relative_coordinates(landmarks, center_idx=24):
    """
    Normalizes coordinates relative to a center point (e.g., hip center).
    
    Args:
        landmarks: List or array of [x, y, z] coordinates.
        center_idx: Index of the landmark to use as the center (default 24: Right Hip).
                   Note: A better center might be the average of 23 (Left Hip) and 24 (Right Hip).
        
    Returns:
        np.array: Relative coordinates.
    """
    landmarks = np.array(landmarks)
    center_point = landmarks[center_idx]
    return landmarks - center_point
