"""Our 5-class MI dataset for MOABB.

This module wraps our TMSi APEX 5-class MI dataset into a
MOABB-compatible format for integration with other datasets.

Example:
    from data_proc import Our5Class
    from moabb.paradigms import MotorImagery
    dataset = Our5Class()
    paradigm = MotorImagery(events=dataset.event_id, n_classes=5)
    X, labels, meta = paradigm.get_data(datasets=dataset, subjects=[1])
"""

import logging
import os
from pathlib import Path
from csv import DictReader

import mne
from moabb.datasets.base import BaseDataset

from .poly5_reader import Poly5Reader

log = logging.getLogger(__name__)


class Our5Class(BaseDataset):
    """Our 5-class MI dataset recorded with TMSi APEX.

    Dataset contains EEG recordings from 14 subjects performing 5 different motor imagery tasks:
    left hand, right hand, left leg, right leg, and tongue movement.

    The experiment protocol was as follows: a fixation cross appeared, followed by a visual cue,
    indicating the target movement. Then "Imagine the movement" text appeared (marking trial start),
    and, finally, "Rest" text appeared (marking trial end). The MI period lasted about 6 seconds.

    EEG was recorded using TMSi APEX with 22 channels at 1000 Hz sampling rate.
    Channels follow the standard 10-20 montage with M1/M2 mastoid references.

    Each subject has multiple recording runs (poly5 files) with corresponding marker files
    indicating trial timings and labels.

    Attributes:
        CHANNELS: List of EEG channel names used in the recordings.
        REF_CHANNELS: Reference channels for re-referencing.
        MONTAGE: Standard montage for channel positions.
        DEFAULT_SFREQ: Original sample frequency in Hz.
    """

    CHANNELS = ["Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
                     "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "POz", "O1", "Oz", "O2"]
    _CH_TYPES = 24 * ["eeg"] + 1 * ["misc"] + 4 * ["misc"]  # 24 eeg, 1 trigger, 4 misc channels
    REF_CHANNELS = ["M1", "M2"]
    MONTAGE = "standard_1020"
    DEFAULT_SFREQ = 1000

    # Marker timestamps are relative to experiment start, but each .poly5 recording has a
    # variable-length preamble (the get-FPS phase) at its front. The experiment occupies
    # exactly the last EXPERIMENT_DURATION_S seconds of the file, so we crop the preamble
    # before attaching annotations to align marker t=0 with raw sample 0.
    EXPERIMENT_DURATION_S = 600.0
    EXPERIMENT_DURATION_S_OVERRIDES = {14: 767.55}

    def __init__(self, data_path="our_data/our_5_class", get_last_trials_xor=None):
        """
        Args:
            data_path: Path to the root dir containing the subject dirs.
                Each subject dir should contain .poly5 files and corresponding _markers.csv files.
            get_last_trials_xor: Defaults to None, including all trials of all selected subjects;
                True for only fetching the last trial per subject; False for all but the last trial per subject.
        """

        super().__init__(
            subjects=list(range(1, 15)),
            sessions_per_subject=1,  # This denotes min sessions per subject
            events={"left_hand": 1, "right_hand": 2, "left_leg": 3, "right_leg": 4, "tongue": 5},
            code="Our5Class",
            interval=[0, 6],  # MI period from "Imagine" to "Rest" (~6s)
            paradigm="imagery",
            doi=""
        )
        self._data_path = Path(data_path)
        self._get_last_trials_xor = get_last_trials_xor

    def _get_single_subject_data(self, subject) -> dict:
        """Load and preprocess data for a single subject.

        Reads all poly5 files for the given subject, extracts trial markers,
        and constructs continuous MNE Raw objects with event annotations.

        Args:
            subject: Subject ID (1-indexed for moabb :cry).

        Returns:
            Dictionary with session keys mapping to run dictionaries,
            each containing an MNE Raw object. Structure:
            {"session_0": {"run_0": Raw, ...}, ...}
        """
        subject_dir = self._get_subject_dir(subject)
        poly5_files = sorted(subject_dir.rglob("*.poly5"))

        # XOR get last trials if set
        if self._get_last_trials_xor is not None:
            poly5_files = poly5_files[-1:] if self._get_last_trials_xor else poly5_files[:-1]

        if not poly5_files:
            raise FileNotFoundError(f"No poly5 files found for subject {subject} in {subject_dir}")

        sessions = dict()
        session_key = "0"  # Single session (subject doesn't take the headset off), multiple runs
        sessions[session_key] = dict()

        experiment_duration_s = self.EXPERIMENT_DURATION_S_OVERRIDES.get(
            subject, self.EXPERIMENT_DURATION_S)

        for run_idx, poly5_path in enumerate(poly5_files):
            raw = self._load_single_run(poly5_path, experiment_duration_s)
            sessions[session_key][str(run_idx)] = raw

        return sessions

    def _load_single_run(self, poly5_path: Path, experiment_duration_s: float) -> mne.io.RawArray:
        """Load and preprocess a single recording run.

        Args:
            poly5_path: Path to the poly5 file.
            experiment_duration_s: Expected duration of the experiment in seconds. The file's
                leading preamble (file duration - experiment_duration_s) is cropped off so
                marker timestamps (experiment-relative) align with raw sample 0.

        Returns:
            MNE Raw object with proper channel types, montage, and event annotations added.
        """
        # Suppress Poly5Reader verbose output
        with open(os.devnull, 'w') as devnull:
            import sys
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                reader = Poly5Reader(str(poly5_path))
                raw = reader.read_data_mne()
            finally:
                sys.stdout = old_stdout

        # Set channel types and montage
        ch_type_mapping = {name: self._CH_TYPES[i] for i, name in enumerate(raw.info["ch_names"])}
        raw.set_channel_types(ch_type_mapping)

        raw.info.set_montage(self.MONTAGE)

        raw.pick(picks=self.CHANNELS + self.REF_CHANNELS)
        raw.set_eeg_reference(ref_channels=self.REF_CHANNELS)
        raw.drop_channels(self.REF_CHANNELS)

        # Crop preamble so marker t=0 aligns with sample 0
        file_duration_s = raw.n_times / raw.info["sfreq"]
        crop_tmin = file_duration_s - experiment_duration_s
        if crop_tmin < 0:
            raise ValueError(
                f"File {poly5_path.name} is shorter ({file_duration_s:.2f}s) than expected "
                f"experiment duration ({experiment_duration_s:.2f}s)")
        raw.crop(tmin=crop_tmin)

        # Load markers as annotations
        markers = self._load_markers(poly5_path)
        if markers:
            self._add_annotations(raw, markers)

        return raw

    def _load_markers(self, poly5_path: Path) -> list[dict]:
        """Load trial markers from CSV corresponding to a poly5 file.

        Args:
            poly5_path: Path to the poly5 file. Marker file is expected to have the same stem
                with "_markers" suffix.

        Returns:
            List of marker dictionaries with keys: time_start_s, time_end_s, body_part.
        """
        marker_path = poly5_path.with_stem(poly5_path.stem + "_markers").with_suffix(".csv")

        if not marker_path.exists():
            log.warning(f"Marker file not found: {marker_path}")
            return []

        with open(marker_path, newline="", encoding="utf-8") as f:
            reader = DictReader(f)
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
            return list(reader)

    def _add_annotations(self, raw: mne.io.RawArray, markers: list[dict]):
        """Add trial markers as MNE annotations.

        MOABB paradigms expect events to be annotated in a stim channel or as annotations.
        We use annotations.

        Args:
            raw: MNE Raw object to add annotations to (modified in-place).
            markers: List of marker dicts.
        """
        onsets = []
        durations = []
        descriptions = []

        for marker in markers:
            try:
                onset = float(marker["time_start_s"])
                end = float(marker["time_end_s"])
                duration = end - onset
                body_part = marker["body_part"]
                event_name = body_part.lower().replace(" ", "_") # MOABB event naming convention

                onsets.append(onset)
                durations.append(duration)
                descriptions.append(event_name)
            except (KeyError, ValueError) as e:
                log.warning(f"Skipping invalid marker: {marker}, error: {e}")

        if onsets:
            annotations = mne.Annotations(
                onset=onsets,
                duration=durations,
                description=descriptions
            )
            raw.set_annotations(annotations)


    def _get_subject_dir(self, subject: int) -> Path:
        """Get the dir path for a subject.

        Args:
            subject: Subject ID (1-indexed for moabb :cry).

        Returns:
            Path to the subject's data dir.
        """
        return self._data_path / f"subject{subject}"

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ) -> list[str | Path]:
        """Return path to local copy of subject data.

        This is a local dataset, no downloading.

        Args:
            subject: Subject ID (1-indexed for moabb :cry).
            path: Unused, kept for API compatibility.
            force_update: Unused, kept for API compatibility.
            update_path: Unused, kept for API compatibility.
            verbose: Unused, kept for API compatibility.

        Returns:
            List containing the path to subject's data dir.

        Raises:
            ValueError: if subject ID is not in subject_list.
            FileNotFoundError: if subject dir does not exist.
        """
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject {subject}. Valid: {self.subject_list}")

        subject_dir = self._get_subject_dir(subject)

        if not subject_dir.exists():
            raise FileNotFoundError(f"Subject dir not found: {subject_dir}."
                                    f"Please ensure data_path is set correctly.")

        return [subject_dir]