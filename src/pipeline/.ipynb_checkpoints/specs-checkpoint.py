from dataclasses import dataclass

@dataclass
class ExperimentSpec:
    """
    Container describing a single experiment run configuration.
    This dataclass stores the metadata needed to locate the input data (video frames and
    COSMED reference file) and to label results during analysis.

    Args:
        marker (str): Run identifier / label in COSMED data.
        camera (str): Camera identifier ("olympus", "flir", "gray", "phone").
        fps (float): Frame rate of the video sequence in Hz.
        folder (str): Path to the folder containing the experiment frames/data.
        cosmed_path (str): Path to the corresponding COSMED reference file.
        subject (str): Subject identifier.
    """
    marker: str
    camera: str
    fps: float
    folder: str
    cosmed_path: str
    subject: str