from pathlib import Path

import torch
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.face_recognition.core.facekit.embedder_mobilefacenet_pytorch import MobileFaceNet


class Command(BaseCommand):
    help = "Strip non-backbone weights from MFNet checkpoint and save lightweight model in core/models."

    def add_arguments(self, parser):
        parser.add_argument(
            "--input",
            type=str,
            default="",
            help="Input checkpoint path. Default: FACE_MODEL_PATH from settings.",
        )
        parser.add_argument(
            "--output",
            type=str,
            default="MFNet_backbone_only.pth",
            help="Output filename or absolute path.",
        )

    def handle(self, *args, **options):
        default_input = getattr(
            settings,
            "FACE_MODEL_PATH",
            r"D:\PythonWorkspace\PBL5-Model\Recognition\checkpoint\MFNet.pth",
        )
        input_arg = (options.get("input") or "").strip()
        input_path = Path(input_arg or default_input)
        if not input_path.is_absolute():
            input_path = Path(settings.BASE_DIR) / input_path
        input_path = input_path.resolve()

        if not input_path.exists():
            raise CommandError(f"Input checkpoint not found: {input_path}")

        output_arg = (options.get("output") or "").strip()
        models_dir = Path(settings.BASE_DIR) / "apps" / "face_recognition" / "core" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        output_path = Path(output_arg)
        if not output_path.is_absolute():
            output_path = models_dir / output_path
        output_path = output_path.resolve()

        checkpoint = torch.load(str(input_path), map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise CommandError("Unsupported checkpoint format. Expected dict-like object.")

        state_dict = (
            checkpoint.get("state_dict")
            or checkpoint.get("model_state_dict")
            or checkpoint.get("backbone_state_dict")
            or checkpoint
        )
        if not isinstance(state_dict, dict):
            raise CommandError("Could not extract state_dict from checkpoint.")

        normalized_state = {}
        for key, value in state_dict.items():
            normalized_key = key
            for prefix in ("module.", "model.", "backbone.", "net."):
                if normalized_key.startswith(prefix):
                    normalized_key = normalized_key[len(prefix):]
            normalized_state[normalized_key] = value

        model = MobileFaceNet(emb=512)
        reference = model.state_dict()
        backbone_only = {
            key: value
            for key, value in normalized_state.items()
            if key in reference and hasattr(value, "shape") and tuple(value.shape) == tuple(reference[key].shape)
        }

        if not backbone_only:
            raise CommandError("No compatible MobileFaceNet backbone weights found.")

        torch.save({"state_dict": backbone_only}, str(output_path))

        self.stdout.write(self.style.SUCCESS("Backbone checkpoint exported successfully."))
        self.stdout.write(f"Input:  {input_path}")
        self.stdout.write(f"Output: {output_path}")
        self.stdout.write(f"Kept tensors: {len(backbone_only)}/{len(reference)}")
