"""Poly5 File Reader.

This module provides functionality to read and process Poly5 files from
Twente Medical Systems International B.V.

Copyright (c) 2022-2024 Twente Medical Systems International B.V.
Licensed under the Apache License, Version 2.0.
"""

import datetime
import locale
import struct
# import tkinter as tk
# from tkinter import filedialog

import mne
import numpy as np
import pandas as pd


class Poly5Reader:
    """Reads and processes Poly5 format files.

    This class handles reading Poly5 files, extracting signal data, and
    converting to various formats including MNE RawArray and CSV.

    Attributes:
        filename: Path to the Poly5 file.
        read_all: Whether to read all data blocks immediately.
        sample_rate: Sampling frequency in Hz.
        num_channels: Number of channels in the recording.
        num_samples: Total number of samples.
        start_time: Recording start timestamp.
        num_data_blocks: Number of data blocks in the file.
        num_samples_per_block: Samples per data block.
        channels: List of Channel objects.
        samples: Numpy array containing signal data.
        ch_names: List of channel names.
        ch_unit_names: List of channel unit names.
    """

    def __init__(self, filename, read_all=True):
        """Initializes the Poly5Reader.

        Args:
            filename: Path to the Poly5 file.
            read_all: If True, reads all data blocks immediately.
        """
        # if filename is None:
        #     root = tk.Tk()
        #     filename = filedialog.askopenfilename(
        #         title="Select poly5-file",
        #         filetypes=(("poly5-files", "*.poly5"), ("All files", "*.*"))
        #     )
        #     root.withdraw()

        self.filename = filename
        self.read_all = read_all
        print("Reading file", filename)
        self._read_file(filename)

    def read_data_mne(self, add_ch_locs=False) -> mne.io.RawArray:
        """Returns MNE RawArray with channel names and types.

        Converts the internal signal data to an MNE RawArray object,
        automatically detecting channel types based on naming conventions.

        Args:
            add_ch_locs: Whether to add channel locations (currently unused).

        Returns:
            An MNE RawArray object containing the signal data.
        """
        fs = self.sample_rate
        labels = self.ch_names
        units = self.ch_unit_names

        type_options = [
            "ecg", "bio", "stim", "eog", "misc", "seeg", "dbs", "ecog",
            "mag", "eeg", "ref_meg", "grad", "emg", "hbr", "hbo"
        ]
        types_clean = []
        for idx, label in enumerate(labels):
            for t_option in type_options:
                if t_option in label.lower():
                    types_clean.append(t_option)
                    break
            else:
                if 'V' in units[idx]:
                    types_clean.append("eeg")
                else:
                    types_clean.append("misc")

        info = mne.create_info(
            ch_names=labels, sfreq=fs, ch_types=types_clean
        )

        # Convert from microvolts to volts if necessary.
        scale = np.array([
            1e-6 if (u == "µVolt" or u == "uVolt" or u == '\u03BCVolt') else 1
            for u in units
        ])

        raw = mne.io.RawArray(
            self.samples * np.expand_dims(scale, axis=1), info
        )
        return raw

    def export_to_csv(self):
        """Exports signal data to CSV format.

        Creates a CSV file with channel names, units, and sample data.
        The sample rate is added as an additional column.
        """
        # Add unit names to the column header.
        ch_names = [
            self.ch_names[i] + " (" + self.ch_unit_names[i] + ")"
            for i in range(len(self.ch_names))
        ]
        ch_names += ["Fs (Hz)"]

        # Add sample rate as a separate column to the samples vector.
        samples = np.vstack(
            (self.samples, np.zeros((1, np.shape(self.samples)[1])))
        )
        samples[-1, :] = self.sample_rate

        # Get decimal point representation (in local language settings).
        lang_locale = locale.getdefaultlocale()[0]
        locale.setlocale(locale.LC_ALL, lang_locale)
        dp = locale.localeconv()["decimal_point"]

        # Write to dataframe.
        df = pd.DataFrame(data=samples.T, columns=ch_names)

        # Export dataframe to .csv.
        if self.filename.lower().endswith(".poly5"):
            save_name = self.filename.lower().replace(".poly5", ".csv")
            df.to_csv(
                path_or_buf=save_name,
                sep=';',
                decimal=dp,
                index=False,
                encoding="utf-16"
            )
            print("Exported to .csv successfully")

    def read_samples(self, n_blocks=None):
        """Reads a subset of sample blocks from the file.

        Args:
            n_blocks: Number of blocks to read. If None, reads all blocks.

        Returns:
            Numpy array containing the sample data.
        """
        if n_blocks is None:
            n_blocks = self.num_data_blocks

        sample_buffer = np.zeros(
            self.num_channels * n_blocks * self.num_samples_per_block
        )

        for i in range(n_blocks):
            data_block = self._read_signal_block(
                self.file_obj, self._buffer_size, self._myfmt
            )
            i1 = i * self.num_samples_per_block * self.num_channels
            i2 = (i + 1) * self.num_samples_per_block * self.num_channels
            sample_buffer[i1:i2] = data_block

        samples = np.transpose(
            np.reshape(
                sample_buffer,
                [self.num_samples_per_block * (i + 1), self.num_channels]
            )
        )
        return samples

    def read_live_impedance(self):
        """Reads live measured impedances stored in the datafile.

        Extracts real and imaginary parts of impedance measurements if
        they are stored in the file.

        Returns:
            A tuple containing:
                - live_imp: Real part of impedances for all channels.
                - live_cap: Imaginary part of impedances for all channels.
                Returns empty lists if no live impedances are stored.
        """
        samples = self.samples
        ch_names = self.ch_names
        live_imp_in_file = False

        num_channels = len(ch_names)

        # Find the channels in which the information is stored.
        for i in range(len(ch_names)):
            # Find channel with ID information.
            if ch_names[i] == "CYCL_IDX":
                cycl_idx_num = i
                live_imp_in_file = True
            # In channel CYCL_ST1 the real part of the impedance value is stored.
            elif ch_names[i] == "CYCL_ST1":
                cycl_imp_num = i
                live_imp_in_file = True
            # In channel CYCL_ST2 the imaginary part of the impedance is stored.
            elif ch_names[i] == "CYCL_ST2":
                cycl_cap_num = i
                live_imp_in_file = True

        # Do not proceed if there are no impedance values stored.
        if not live_imp_in_file:
            print("No live impedances were stored in this file")
            return [], []

        # Cycle_idx channel defines the channel index of which the impedance
        # information is stored in channels CYCL_ST1 and CYCL_ST2.
        maximum_cycl_idx = np.max(samples[cycl_idx_num, :])

        # Last index of the channel information is maximum_cylc_idx.
        # Channel indices range from 0 to this number.
        length_stored_idx = int(maximum_cycl_idx + 1)

        cycl_idx = samples[cycl_idx_num, :]
        cycl_imp = samples[cycl_imp_num, :]
        cycl_cap = samples[cycl_cap_num, :]

        # Define variables that store live impedance and live cap per channel.
        # By default, set the values to 1000.
        live_imp = np.ones(
            (num_channels, len(samples[cycl_imp_num, :]))
        ) * 1000
        live_cap = np.ones(
            (num_channels, len(samples[cycl_cap_num, :]))
        ) * 1000

        # Loop through the channels and store the values.
        for i in range(len(cycl_idx)):
            # Values are measured once in every length_stored_idx channels.
            live_imp[int(cycl_idx[i]), i:i + length_stored_idx] = cycl_imp[i]
            live_cap[int(cycl_idx[i]), i:i + length_stored_idx] = cycl_cap[i]

        return live_imp[:, :], live_cap[:, :]

    def close(self):
        """Closes the file object."""
        self.file_obj.close()

    def _read_file(self, filename):
        """Reads the entire Poly5 file.

        Args:
            filename: Path to the Poly5 file.
        """
        try:
            self.file_obj = open(filename, "rb")
            file_obj = self.file_obj
            try:
                self._read_header(file_obj)
                self.channels = self._read_signal_description(file_obj)
                self._myfmt = 'f' * self.num_channels * self.num_samples_per_block
                self._buffer_size = self.num_channels * self.num_samples_per_block

                # if self.read_all:
                #     sample_buffer = np.zeros(
                #         self.num_channels * self.num_samples
                #     )
                #
                #     for i in range(self.num_data_blocks):
                #         if i % 100 == 0:
                #             print(f"\rProgress: {100 * i / self.num_data_blocks:.1f}%", end="\r")
                #
                #         # Check whether final data block is filled completely.
                #         if i == self.num_data_blocks - 1:
                #             final_block_size = (
                #                 self.num_samples / self.num_data_blocks
                #             )
                #             if final_block_size % self.num_samples_per_block != 0:
                #                 remaining_samples = (
                #                     self.num_samples % self.num_samples_per_block
                #                 )
                #                 data_block = self._read_signal_block(
                #                     file_obj,
                #                     buffer_size=remaining_samples * self.num_channels,
                #                     myfmt='f' * remaining_samples * self.num_channels
                #                 )
                #             else:
                #                 data_block = self._read_signal_block(
                #                     file_obj, self._buffer_size, self._myfmt
                #                 )
                #         else:
                #             data_block = self._read_signal_block(
                #                 file_obj, self._buffer_size, self._myfmt
                #             )
                #
                #         # Get indices that need to be filled in the samples array.
                #         i1 = i * self.num_samples_per_block * self.num_channels
                #         i2 = (i + 1) * self.num_samples_per_block * self.num_channels
                #
                #         # Correct for final data block if not fully filled.
                #         if i2 >= self.num_samples * self.num_channels:
                #             i2 = self.num_samples * self.num_channels
                #
                #         sample_buffer[i1:i2] = data_block
                #
                #     samples = np.transpose(
                #         np.reshape(
                #             sample_buffer,
                #             [self.num_samples, self.num_channels]
                #         )
                #     )

                # vvv TRYING OUT AN OPTIMIZATION vvv
                if self.read_all:
                    # Calculate total data size
                    header_size_per_block = 86
                    data_size_per_block = self.num_channels * self.num_samples_per_block * 4
                    block_size = header_size_per_block + data_size_per_block

                    # Pre-allocate array
                    sample_buffer = np.zeros(self.num_channels * self.num_samples, dtype=np.float32)

                    for i in range(self.num_data_blocks):
                        if i % 100 == 0:
                            print(f"\rProgress: {100 * i / self.num_data_blocks:.1f}%", end="\r")

                        # Skip 86-byte block header
                        file_obj.read(86)

                        # Handle final block if partially filled
                        if i == self.num_data_blocks - 1:
                            remaining = self.num_samples - i * self.num_samples_per_block
                            samples_to_read = remaining * self.num_channels
                        else:
                            samples_to_read = self.num_channels * self.num_samples_per_block

                        # Read directly into numpy
                        data = np.frombuffer(file_obj.read(samples_to_read * 4), dtype=np.float32)

                        i1 = i * self.num_samples_per_block * self.num_channels
                        i2 = i1 + len(data)
                        sample_buffer[i1:i2] = data
                        # ^^^ END OPTIMIZATION ^^^

                    samples = np.transpose(
                        np.reshape(
                            sample_buffer,
                            [self.num_samples, self.num_channels]
                        )
                    )

                    ch_names = [s._Channel__name for s in self.channels]
                    self.ch_unit_names = [
                        s._Channel__unit_name for s in self.channels
                    ]

                    self.samples, self.ch_names = self._reorder_grid(
                        samples, ch_names
                    )

                    print("Done reading data.")
                    self.file_obj.close()

            except Exception as e:
                print("Reading data failed, because of the following error:\n")
                raise
        except OSError:
            print("Could not open file.")

    def _read_header(self, f):
        """Reads the file header.

        Args:
            f: File object to read from.
        """
        header_data = struct.unpack(
            "=31sH81phhBHi4xHHHHHHHiHHH64x", f.read(217)
        )
        magic_number = str(header_data[0])
        version_number = header_data[1]
        self.sample_rate = header_data[3]
        self.num_channels = header_data[6] // 2
        self.num_samples = header_data[7]
        self.start_time = datetime.datetime(
            header_data[8], header_data[9], header_data[10],
            header_data[12], header_data[13], header_data[14]
        )
        self.num_data_blocks = header_data[15]
        self.num_samples_per_block = header_data[16]

        if magic_number != "b'POLY SAMPLE FILEversion 2.03\\r\\n\\x1a'":
            print("This is not a Poly5 file.")
        elif version_number != 203:
            print("Version number of file is invalid.")
        else:
            print("\t Number of samples:  %s" % self.num_samples)
            print("\t Number of channels:  %s" % self.num_channels)
            print("\t Sample rate: %s Hz" % self.sample_rate)

    def _read_signal_description(self, f):
        """Reads signal channel descriptions.

        Args:
            f: File object to read from.

        Returns:
            List of Channel objects.
        """
        chan_list = []
        for ch in range(self.num_channels):
            channel_description = struct.unpack(
                "=41p4x11pffffH62x", f.read(136)
            )
            name = channel_description[0][5:].decode("ascii")
            unit_name = channel_description[1].decode("utf-8")
            ch = Channel(name, unit_name)
            chan_list.append(ch)
            f.read(136)
        return chan_list

    def _read_signal_block(self, f, buffer_size, myfmt):
        """Reads a single signal data block.

        Args:
            f: File object to read from.
            buffer_size: Size of the buffer to read.
            myfmt: Struct format string for unpacking.

        Returns:
            Numpy array containing the signal block data.
        """
        f.read(86)
        sample_data = f.read(buffer_size * 4)

        # data_block = struct.unpack(myfmt, sample_data)
        # signal_block = np.asarray(data_block)
        signal_block = np.frombuffer(sample_data, dtype=np.float32)  # this is like 37% faster

        return signal_block

    def _reorder_grid(self, samples, ch_names):
        """Reorders textile grid channels.

        Args:
            samples: Numpy array of sample data.
            ch_names: List of channel names.

        Returns:
            Tuple of (reordered_samples, reordered_ch_names).
        """
        channel_conversion_list = np.arange(0, len(ch_names), dtype=int)

        # Detect row and column number based on channel name.
        rc_ch = []
        for i, ch in enumerate(ch_names):
            if ch.find('R') == 0 and ch.find('C') == 2:
                r, c = ch[1:].split('C')
                rc_ch.append((r, str(c).zfill(2), i))
            elif ch == "CREF":
                rc_ch.append(('0', '0', i))

        # Sort data based on row and column.
        rc_ch.sort()
        for ch in range(len(rc_ch)):
            channel_conversion_list[ch] = rc_ch[ch][2]

        # Change the ordering of channels on the textile grid.
        samples = samples[channel_conversion_list, :]
        ch_names = [ch_names[i] for i in channel_conversion_list]

        return samples, ch_names


class Channel:
    """Represents a device channel.

    Attributes:
        name: The name of the channel.
        unit_name: The name of the unit (e.g. 'μVolt') of the sample data.
    """

    def __init__(self, name, unit_name):
        """Initializes a Channel.

        Args:
            name: The channel name.
            unit_name: The unit name for the channel's data.
        """
        self.__unit_name = unit_name
        self.__name = name


if __name__ == "__main__":
    data = Poly5Reader()
