import json
from pathlib import Path

from nemo.collections.asr.models import ClusteringDiarizer
from omegaconf import OmegaConf

CONFIG = OmegaConf.load(Path(__file__).parent / "diar_config.yaml")
MERGE_GAP = 1.0  # join same-speaker turns separated by less than this many seconds


def diarize(wav: Path, work_dir: Path, num_speakers: int | None, max_speakers: int, speaker_model=None) -> list[dict]:
    """Return speaker turns [{start, end, speaker}] for a 16 kHz mono wav."""
    manifest = work_dir / "manifest.json"
    manifest.write_text(json.dumps({
        "audio_filepath": str(wav), "offset": 0, "duration": None, "label": "infer", "text": "-",
        "num_speakers": num_speakers, "rttm_filepath": None, "uem_filepath": None,
    }) + "\n")

    cfg = CONFIG.copy()
    cfg.diarizer.manifest_filepath = str(manifest)
    cfg.diarizer.out_dir = str(work_dir)
    cfg.diarizer.clustering.parameters.oracle_num_speakers = num_speakers is not None
    cfg.diarizer.clustering.parameters.max_num_speakers = max_speakers
    ClusteringDiarizer(cfg=cfg, speaker_model=speaker_model).diarize()

    turns = []
    for line in (work_dir / "pred_rttms" / f"{wav.stem}.rttm").read_text().splitlines():
        f = line.split()
        start, dur, spk = float(f[3]), float(f[4]), "SPEAKER_" + f[7].split("_")[-1]
        if turns and turns[-1]["speaker"] == spk and start - turns[-1]["end"] < MERGE_GAP:
            turns[-1]["end"] = start + dur
        else:
            turns.append({"start": start, "end": start + dur, "speaker": spk})
    return turns
