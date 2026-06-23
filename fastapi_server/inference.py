# ============================================================
# ONNX INFERENCE ENGINE
# ============================================================

import os
from pathlib import Path
import onnxruntime as ort
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

# ============================================================
# ONNX INFERENCE CLASS
# ============================================================

class ONNXInference:
    """
    ONNX inference engine for UNet++ boundary prediction
    """

    def __init__(self, model_path, providers=None, tile_size=768, stride=384, blend_sigma=None):
        """
        Initialize ONNX inference session

        Parameters:
        - model_path: Path to ONNX model
        - providers: ONNX Runtime execution providers
        - tile_size: Patch size for overlap inference
        - stride: Step size between overlapping patches
        - blend_sigma: Gaussian sigma used for blending
        """

        if providers is None:
            providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider"
            ]

        self.tile_size = tile_size
        self.stride = stride
        self.blend_sigma = blend_sigma if blend_sigma is not None else tile_size / 8.0

        logger.info(f"Loading ONNX model from: {model_path}")

        try:
            self.session = ort.InferenceSession(
                model_path,
                providers=providers
            )

            logger.info(f"Available providers: {ort.get_available_providers()}")
            logger.info(f"Using providers: {self.session.get_providers()}")

            # Get input/output info
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            self.input_shape = self.session.get_inputs()[0].shape
            self.output_shape = self.session.get_outputs()[0].shape

            logger.info(f"Input shape: {self.input_shape}")
            logger.info(f"Output shape: {self.output_shape}")

        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise

    def preprocess(self, image, target_h=None, target_w=None):
        """
        Preprocess image for model inference

        Parameters:
        - image: Input image (BGR from OpenCV)
        - target_h: Target height for model input
        - target_w: Target width for model input

        Returns:
        - Preprocessed image tensor
        """

        target_h = target_h or self.tile_size
        target_w = target_w or self.tile_size

        resized = cv2.resize(
            image,
            (int(target_w), int(target_h)),
            interpolation=cv2.INTER_LINEAR
        )

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        normalized = rgb.astype(np.float32) / 255.0

        tensor = np.transpose(normalized, (2, 0, 1))
        tensor = np.expand_dims(tensor, 0).astype(np.float32)

        return tensor

    def postprocess(self, output, original_height, original_width):
        """
        Postprocess model output

        Parameters:
        - output: Model output tensor
        - original_height: Original image height
        - original_width: Original image width

        Returns:
        - Segmentation mask (0-1)
        """

        mask = output.squeeze(0)

        if mask.ndim == 3:
            mask = mask[0]

        mask = np.clip(mask, 0.0, 1.0)

        mask_resized = cv2.resize(
            mask,
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR
        )

        return mask_resized

    def _get_tile_positions(self, dimension):
        if dimension <= self.tile_size:
            return [0]

        positions = list(range(0, dimension - self.tile_size + 1, self.stride))
        if positions[-1] != dimension - self.tile_size:
            positions.append(dimension - self.tile_size)

        return positions

    def _pad_patch(self, patch):
        patch_h, patch_w = patch.shape[:2]
        pad_h = max(0, self.tile_size - patch_h)
        pad_w = max(0, self.tile_size - patch_w)

        if pad_h == 0 and pad_w == 0:
            return patch, 0, 0

        padded = cv2.copyMakeBorder(
            patch,
            0,
            pad_h,
            0,
            pad_w,
            cv2.BORDER_REFLECT_101
        )

        return padded, pad_h, pad_w

    def _create_weight_map(self):
        gaussian_1d = cv2.getGaussianKernel(self.tile_size, self.blend_sigma)
        weight = gaussian_1d @ gaussian_1d.T
        weight = weight.astype(np.float32)
        weight /= weight.max()
        return weight

    def _save_debug_image(self, output_dir, filename, image):
        if output_dir is None:
            return None

        output_path = Path(output_dir) / filename

        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating):
                image = np.clip(image, 0.0, 1.0)
                image = (image * 255.0).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)

        cv2.imwrite(str(output_path), image)
        return str(output_path)

    def _infer_patch(self, patch):
        input_tensor = self.preprocess(patch)
        output = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )[0]
        return self.postprocess(output, patch.shape[0], patch.shape[1])

    def predict(self, image, debug_output_dir=None):
        original_h, original_w = image.shape[:2]
        mask, debug_info = self.predict_with_overlap(
            image,
            debug_output_dir=debug_output_dir
        )

        if debug_output_dir is not None:
            self._save_debug_image(debug_output_dir, 'debug_prediction.png', mask)

        return mask, debug_info

    def predict_with_overlap(self, image, debug_output_dir=None):
        image_h, image_w = image.shape[:2]

        positions_y = self._get_tile_positions(image_h)
        positions_x = self._get_tile_positions(image_w)

        merged = np.zeros((image_h, image_w), dtype=np.float32)
        normalizer = np.zeros((image_h, image_w), dtype=np.float32)
        weight_map = self._create_weight_map()

        patch_files = []
        patch_idx = 0

        for y in positions_y:
            for x in positions_x:
                patch = image[y:y + self.tile_size, x:x + self.tile_size]
                patched, pad_h, pad_w = self._pad_patch(patch)

                prediction = self._infer_patch(patched)

                if pad_h > 0 or pad_w > 0:
                    prediction = prediction[:patch.shape[0], :patch.shape[1]]

                patch_h, patch_w = prediction.shape[:2]
                weight = weight_map[:patch_h, :patch_w]

                merged[y:y + patch_h, x:x + patch_w] += prediction * weight
                normalizer[y:y + patch_h, x:x + patch_w] += weight

                if debug_output_dir is not None and patch_idx < 12:
                    debug_file = self._save_debug_image(
                        debug_output_dir,
                        f'debug_patch_{patch_idx + 1}.png',
                        prediction
                    )
                    if debug_file:
                        patch_files.append(debug_file)

                patch_idx += 1

        valid = normalizer > 1e-6
        merged[valid] /= normalizer[valid]

        if not np.all(valid):
            merged[~valid] = 0.0

        debug_info = {
            'patch_files': patch_files,
            'num_patches': patch_idx
        }

        return merged, debug_info
