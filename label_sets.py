import os
from collections import namedtuple
from pathlib import Path
from typing import Union, Dict

from torchvideo.datasets import LabelSet, Label


FileListEntry = namedtuple("FileListEntry", ["path", "num_frames", "label"])


class FileList:
    def __init__(self, filelist_path: Union[str, Path]):
        self.filelist_path = Path(filelist_path)
        self.filelist = self._read_filelist(self.filelist_path)

    def __getitem__(self, video_name: str) -> FileListEntry:
        return self.filelist[video_name]

    def __contains__(self, video_name: str):
        return video_name in self.filelist

    @classmethod
    def _read_filelist(cls, filelist_path: Path) -> Dict[str, FileListEntry]:
        filelist = {}
        with filelist_path.open('r') as f:
            for line in f:
                entry = cls._parse_filelist_line(line)
                filelist[entry.path.name] = entry
        return filelist

    @staticmethod
    def _parse_filelist_line(line: str) -> FileListEntry:
        # format: "path num_frames label"
        split = line.split(' ')
        label = int(split[-1])
        num_frames = int(split[-2])
        # Rejoin split filepath in case there are any spaces present.
        path = " ".join(split[:-2])
        return FileListEntry(path=Path(path), num_frames=num_frames, label=label)


class FileListLabelSet(LabelSet):
    def __init__(self, filelist: FileList):
        self.filelist = filelist

    def __getitem__(self, video_name: str) -> Label:
        return self.filelist[video_name].label
