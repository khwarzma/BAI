import torch
import torch.onnx
import onnxruntime as rt
import onnx
from pathlib import Path
from typing import Optional
import numpy as np

from model import BaiMicroEncoder


class BaiExporter:
    """Export trained PyTorch model to ONNX with INT8 quantization."""
    
    def __init__(
        self,
        model: BaiMicroEncoder,
        output_path: str = "models/v1.bai",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.eval()
    
    def export_to_onnx(self, onnx_path: Optional[str] = None) -> str:
        """
        Export PyTorch model to ONNX format with dynamic axes.
        
        Args:
            onnx_path: Path to save ONNX model (temporary).
        
        Returns:
            Path to the exported ONNX model.
        """
        if onnx_path is None:
            onnx_path = str(self.output_path).replace(".bai", ".onnx")
        
        # Create dummy inputs
        dummy_input_ids = torch.zeros(1, 256, dtype=torch.long, device=self.device)
        dummy_attention_mask = torch.ones(1, 256, dtype=torch.long, device=self.device)
        
        # Define input and output names
        input_names = ['input_ids', 'attention_mask']
        output_names = ['logits_category', 'logits_otp', 'confidence']
        
        # Define dynamic axes for batch size and sequence length
        dynamic_axes = {
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'logits_category': {0: 'batch_size'},
            'logits_otp': {0: 'batch_size'},
            'confidence': {0: 'batch_size'},
        }
        
        # Export to ONNX with opset_version=17 for C++ compatibility
        torch.onnx.export(
            self.model,
            (dummy_input_ids, dummy_attention_mask),
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
            verbose=False,
        )
        
        print(f"Model exported to ONNX: {onnx_path}")
        return onnx_path
    
    def quantize_onnx(self, onnx_path: str) -> str:
        """
        Apply dynamic INT8 quantization to ONNX model.
        Maintains identical input/output node names for seamless compatibility.
        
        Args:
            onnx_path: Path to the ONNX model.
        
        Returns:
            Path to the quantized model with preserved node names.
        """
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        quantized_path = str(self.output_path)
        
        # Quantize to INT8 (preserves input/output node names)
        quantize_dynamic(
            onnx_path,
            quantized_path,
            weight_type=QuantType.QInt8,
        )
        
        print(f"Model quantized to INT8: {quantized_path}")
        print(f"Input/Output node names preserved for C++ ONNX Runtime compatibility")
        return quantized_path
    
    def validate_export(self, model_path: str) -> bool:
        """
        Validate exported model by running test inference.
        
        Args:
            model_path: Path to the exported model.
        
        Returns:
            True if validation successful, False otherwise.
        """
        try:
            # Create ONNX Runtime session
            session = rt.InferenceSession(
                model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            
            # Prepare test inputs
            test_input_ids = np.random.randint(0, 50257, (1, 128), dtype=np.int64)
            test_attention_mask = np.ones((1, 128), dtype=np.int64)
            
            # Run inference
            outputs = session.run(
                None,
                {
                    'input_ids': test_input_ids,
                    'attention_mask': test_attention_mask,
                }
            )
            
            # Validate outputs
            assert len(outputs) == 3, f"Expected 3 outputs, got {len(outputs)}"
            
            logits_category, logits_otp, confidence = outputs
            
            assert logits_category.shape == (1, 5), \
                f"logits_category shape mismatch: {logits_category.shape}"
            assert logits_otp.shape == (1, 1), \
                f"logits_otp shape mismatch: {logits_otp.shape}"
            assert confidence.shape == (1, 1), \
                f"confidence shape mismatch: {confidence.shape}"
            
            # Validate confidence is in [0, 1]
            assert np.all((confidence >= 0) & (confidence <= 1)), \
                "Confidence values out of range [0, 1]"
            
            print(f"✓ Validation successful!")
            print(f"  - Category logits shape: {logits_category.shape}")
            print(f"  - OTP logits shape: {logits_otp.shape}")
            print(f"  - Confidence shape: {confidence.shape}")
            
            return True
        
        except Exception as e:
            print(f"✗ Validation failed: {e}")
            return False
    
    def export(self, validate: bool = True) -> bool:
        """
        Complete export pipeline: ONNX export → INT8 quantization → validation.
        
        Args:
            validate: Whether to run validation after export.
        
        Returns:
            True if export successful, False otherwise.
        """
        try:
            print("Starting BAI model export pipeline...\n")
            
            # Step 1: Export to ONNX
            onnx_path = self.export_to_onnx()
            
            # Step 2: Quantize to INT8
            quantized_path = self.quantize_onnx(onnx_path)
            
            # Step 3: Validate (optional)
            if validate:
                print("\nValidating exported model...")
                is_valid = self.validate_export(quantized_path)
                if not is_valid:
                    return False
            
            print(f"\n✓ Export pipeline complete!")
            print(f"  Final model: {self.output_path}")
            print(f"  Format: .bai (ONNX INT8 quantized)")
            
            return True
        
        except Exception as e:
            print(f"✗ Export pipeline failed: {e}")
            return False


def export_model(
    checkpoint_path: str,
    output_path: str = "models/v1.bai",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Load trained model from checkpoint and export to .bai format.
    
    Args:
        checkpoint_path: Path to the saved model checkpoint.
        output_path: Path where to save the exported model.
        device: Device to use for export.
    """
    # Initialize model
    model = BaiMicroEncoder(
        d_model=256,
        num_heads=8,
        num_layers=6,
        d_ff=1024,
        max_seq_length=512,
        num_categories=5,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from: {checkpoint_path}")
    
    # Export
    exporter = BaiExporter(
        model=model,
        output_path=output_path,
        device=device,
    )
    
    success = exporter.export(validate=True)
    
    return success


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python export.py <checkpoint_path> [output_path]")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "models/v1.bai"
    
    export_model(checkpoint_path, output_path)
