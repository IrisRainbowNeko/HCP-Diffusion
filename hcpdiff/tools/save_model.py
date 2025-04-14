from diffusers import DiffusionPipeline
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model", default=None, type=str)
parser.add_argument("output", default=None, type=str)
args = parser.parse_args()

pipe = DiffusionPipeline.from_pretrained(args.model, safety_checker=None, requires_safety_checker=False,
                                        resume_download=True)

pipe.save_pretrained(args.output)