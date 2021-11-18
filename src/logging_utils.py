import csv
import dataclasses
import os


def log_artifact(dataclass, instances, logger, epoch, step, path, output_dir):

    filename = f"epoch={epoch}-step={step}"

    with open(os.path.join(output_dir, filename), "w+") as log_file:
        fieldnames = [f.name for f in dataclasses.fields(dataclass)]
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)

        writer.writeheader()

        for i in instances:
            writer.writerow(dataclasses.asdict(i))

    if logger and logger.experiment:
        logger.experiment[path].upload(filename)
