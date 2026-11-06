"""Streaming VR diagnostics; distinct from the camera/replay angle H5 schema."""
import json

import h5py


class VRResultWriter:
    """Keep every input row, including missing/warmup rows and held commands."""

    def __init__(self, path, routes, metadata):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(path, "x")  # Never overwrite a recording or old result.
        try:
            self.file.attrs.update(schema="transhandr_vr_results_v2", angle_units="rad",
                                   timestamp_unit="s", routes=json.dumps(routes), **metadata)
            self.rows = 0
            self.datasets = {}
            specs = {"frame_ids": ((), "i8"), "timestamps": ((), "f8")}
            for side in routes:
                specs.update({f"{side}_{name}": ((), "?")
                              for name in ("input_valid", "model_valid")})
                specs[f"{side}_raw_angles"] = ((18,), "f8")
                specs.update({f"{side}_{name}_positions": ((17,), "f8")
                              for name in ("native", "command")})
            for name, (shape, dtype) in specs.items():
                self.datasets[name] = self.file.create_dataset(
                    name, shape=(0, *shape), maxshape=(None, *shape), dtype=dtype, chunks=True)
        except BaseException:
            self.file.close()
            raise

    def append(self, frame_id, timestamp, sides):
        values = {"frame_ids": frame_id, "timestamps": timestamp}
        for side, data in sides.items():
            values.update({f"{side}_{key}": value for key, value in data.items()})
        for name, dataset in self.datasets.items():
            dataset.resize(self.rows + 1, axis=0)
            dataset[self.rows] = values[name]
        self.rows += 1

    def close(self):
        self.file.close()
