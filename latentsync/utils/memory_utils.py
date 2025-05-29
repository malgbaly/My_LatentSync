import torch
import gc

class MemoryManager:
    @staticmethod
    def clear_gpu_memory():
        """Clear GPU memory cache"""
        torch.cuda.empty_cache()
        gc.collect()
        
    @staticmethod
    def get_gpu_memory_usage():
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2, torch.cuda.memory_reserved() / 1024**2
        return 0, 0
        
    @staticmethod
    def optimize_memory():
        """Optimize memory usage"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            
    @staticmethod
    def set_memory_fraction(fraction):
        """Set memory fraction for GPU"""
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(fraction)
            
    @staticmethod
    def get_optimal_batch_size(model_size_mb, available_memory_mb, safety_factor=0.8):
        """Calculate optimal batch size based on available memory"""
        if not torch.cuda.is_available():
            return 1
            
        # Estimate memory per sample (model size + activations + gradients)
        memory_per_sample = model_size_mb * 3  # Rough estimate
        
        # Calculate maximum batch size
        max_batch_size = int((available_memory_mb * safety_factor) / memory_per_sample)
        return max(1, max_batch_size) 