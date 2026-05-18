"""TMSi APEX 5-class MI dataset wrapped as a MOABB dataset."""

import logging
import os
from pathlib import Path
from csv import DictReader

import mne
from moabb.datasets.base import BaseDataset

from .poly5_reader import Poly5Reader

log = logging.getLogger(__name__)


class Our5Class(BaseDataset):
    """TMSi APEX 5-class MI dataset.

    14 subjects, 22 EEG channels at 1000 Hz raw, 5 MI classes
    (left/right hand, left/right leg, tongue). Each subject has multiple
    poly5 runs with companion `_markers.csv` files. The trial structure
    is fixation cross -> visual cue -> "Imagine the movement" (trial
    start) -> "Rest" (trial end), with a ~6 s MI period.

    Timing:
        Marker timestamps are on the experiment clock, which starts at
        `ExperimentController.start()`. The recorder calls
        `dev.start_measurement(...)` first and then schedules
        `experiment.start` via `QTimer.singleShot(500, ...)`, so the
        file clock leads the experiment clock by `PREAMBLE_S = 0.5` s.
        The raw stays uncropped and every annotation onset is offset by
        `PREAMBLE_S` so that experiment-clock t=0 lands at file-clock
        t=PREAMBLE_S, where the MI actually began. MOABB ignores the
        ~10-15 s trailing tail because it epochs around annotations.
    """

    CHANNELS = ["Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
                     "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "POz", "O1", "Oz", "O2"]
    _CH_TYPES = 24 * ["eeg"] + 1 * ["misc"] + 4 * ["misc"]  # 24 eeg, 1 trigger, 4 misc channels
    REF_CHANNELS = ["M1", "M2"]
    MONTAGE = "standard_1020"
    DEFAULT_SFREQ = 1000

    # Delay between dev.start_measurement and experiment.start in the
    # recorder (QTimer.singleShot(500, experiment.start)). Added to every
    # annotation onset to map experiment-clock onto file-clock seconds.
    PREAMBLE_S = 0.5
    # Soft upper bound on the trailing data after the last trial; longer
    # tails suggest a paused or restarted acquisition.
    _MAX_TAIL_S = 60.0

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

        for run_idx, poly5_path in enumerate(poly5_files):
            raw = self._load_single_run(poly5_path, subject=subject)
            sessions[session_key][str(run_idx)] = raw

        return sessions

    def _load_single_run(self, poly5_path: Path, subject: int) -> mne.io.RawArray:
        """Load and preprocess a single recording run.

        Args:
            poly5_path: Path to the poly5 file.
            subject: Subject id, used only for logging context.

        Returns:
            MNE Raw object with proper channel types, montage, and event annotations added.
            The raw is kept uncropped; marker onsets are offset by ``PREAMBLE_S`` so that
            experiment-clock t=0 maps to file-clock t=PREAMBLE_S.
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

        # Load markers and attach as annotations, offset by PREAMBLE_S so that experiment
        # clock t=0 (the marker origin) maps onto file-clock t=PREAMBLE_S
        markers = self._load_markers(poly5_path)
        self._sanity_log(poly5_path, raw, markers, subject)
        if markers:
            self._add_annotations(raw, markers)

        return raw

    def _sanity_log(self, poly5_path: Path, raw: mne.io.BaseRaw,
                    markers: list[dict], subject: int) -> None:
        """Log per-file timing summary and warn on suspicious recordings.

        Compares file duration against the time of the last marker plus PREAMBLE_S to
        detect truncated recordings (file ends before the last trial) or excessively long
        trailing data (possible paused acquisition or other mishap).

        Args:
            poly5_path: Path to the poly5 file (used for log identification).
            raw: Loaded MNE raw (must already have its preamble untouched).
            markers: Parsed marker list for this run (may be empty).
            subject: Subject id, used only for logging context.
        """
        file_duration_s = raw.n_times / raw.info["sfreq"]
        n_markers = len(markers)
        if not markers:
            log.warning(
                "[Our5Class] subject=%d file=%s file_dur=%.2fs n_markers=0",
                subject, poly5_path.name, file_duration_s)
            return

        try:
            last_end = max(float(m["time_end_s"]) for m in markers)
        except (KeyError, ValueError) as e:
            log.warning("[Our5Class] subject=%d file=%s could not parse last marker: %s",
                        subject, poly5_path.name, e)
            return

        expected_min = self.PREAMBLE_S + last_end
        log.info(
            "[Our5Class] subject=%d file=%s file_dur=%.2fs n_markers=%d "
            "last_marker_end=%.2fs expected_min_dur=%.2fs tail=%.2fs",
            subject, poly5_path.name, file_duration_s, n_markers,
            last_end, expected_min, file_duration_s - expected_min)

        if file_duration_s < expected_min:
            log.warning(
                "[Our5Class] subject=%d file=%s: file truncated, duration %.2fs < "
                "expected %.2fs (last marker refers to data beyond end of file)",
                subject, poly5_path.name, file_duration_s, expected_min)
        elif file_duration_s > expected_min + self._MAX_TAIL_S:
            log.warning(
                "[Our5Class] subject=%d file=%s: excessive trailing data, file_dur=%.2fs "
                "vs expected_min=%.2fs (tail=%.2fs > %.0fs); possible recording mishap",
                subject, poly5_path.name, file_duration_s, expected_min,
                file_duration_s - expected_min, self._MAX_TAIL_S)

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

        Marker `time_start_s` is on the experiment clock; we add `PREAMBLE_S` to land on
        the file clock so that the annotation lines up with the actual MI sample.

        Args:
            raw: MNE Raw object to add annotations to (modified in-place).
            markers: List of marker dicts.
        """
        onsets = []
        durations = []
        descriptions = []

        for marker in markers:
            try:
                onset = float(marker["time_start_s"]) + self.PREAMBLE_S
                end = float(marker["time_end_s"]) + self.PREAMBLE_S
                duration = end - onset
                body_part = marker["body_part"]
                event_name = body_part.lower().replace(" ", "_")  # MOABB event naming convention

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