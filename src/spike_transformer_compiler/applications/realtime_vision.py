"""Real-time vision processing with neuromorphic hardware."""

import time
import threading
import numpy as np
from typing import Dict, Any, Optional, Callable, List, Tuple
import cv2
from queue import Queue, Empty
from ..runtime.executor import NeuromorphicExecutor
from ..logging_config import app_logger
from ..performance import PerformanceProfiler


class VisionPipeline:
    """High-performance vision processing pipeline."""
    
    def __init__(
        self,
        compiled_model: Any,
        input_resolution: Tuple[int, int] = (224, 224),
        preprocessing_steps: Optional[List[str]] = None,
        postprocessing_steps: Optional[List[str]] = None
    ):
        self.compiled_model = compiled_model
        self.input_resolution = input_resolution
        self.preprocessing_steps = preprocessing_steps or ["normalize", "resize"]
        self.postprocessing_steps = postprocessing_steps or ["softmax", "argmax"]
        
        # Performance monitoring
        self.performance_profiler = PerformanceProfiler()
        self.frame_times = []
        self.inference_times = []
        
        # Processing statistics
        self.total_frames = 0
        self.successful_inferences = 0
        self.failed_inferences = 0
        
        app_logger.info(f"VisionPipeline initialized for {input_resolution} resolution")
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess input frame for neuromorphic inference."""
        processed = frame.copy()
        
        for step in self.preprocessing_steps:
            if step == "resize":
                processed = cv2.resize(processed, self.input_resolution)
            elif step == "normalize":
                processed = processed.astype(np.float32) / 255.0
            elif step == "grayscale":
                if len(processed.shape) == 3:
                    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            elif step == "gaussian_blur":
                processed = cv2.GaussianBlur(processed, (5, 5), 0)
            elif step == "edge_detection":
                processed = cv2.Canny(processed.astype(np.uint8), 50, 150)
            elif step == "histogram_equalization":
                if len(processed.shape) == 2:
                    processed = cv2.equalizeHist(processed.astype(np.uint8))
                    
        return processed
        
    def postprocess_output(self, model_output: np.ndarray) -> Dict[str, Any]:
        """Postprocess neuromorphic model output."""
        processed = model_output.copy()
        
        result = {"raw_output": model_output}
        
        for step in self.postprocessing_steps:
            if step == "softmax":
                processed = self._softmax(processed)
                result["probabilities"] = processed
            elif step == "argmax":
                predicted_class = np.argmax(processed)
                result["predicted_class"] = int(predicted_class)
                result["confidence"] = float(processed.max())
            elif step == "threshold":
                threshold = 0.5
                result["binary_output"] = (processed > threshold).astype(int)
            elif step == "top_k":
                k = 5
                top_indices = np.argsort(processed)[-k:][::-1]
                result["top_k_classes"] = top_indices.tolist()
                result["top_k_probabilities"] = processed[top_indices].tolist()
                
        return result
        
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax normalization."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
        
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process single frame through the pipeline."""
        frame_start_time = time.time()
        
        try:
            # Preprocess
            preprocessed = self.preprocess_frame(frame)
            
            # Neuromorphic inference
            inference_start = time.time()
            model_output = self.compiled_model.run(
                preprocessed,
                time_steps=4,
                return_spike_trains=False
            )
            inference_time = time.time() - inference_start
            
            # Postprocess
            result = self.postprocess_output(model_output)
            
            # Update statistics
            frame_time = time.time() - frame_start_time
            self.frame_times.append(frame_time)
            self.inference_times.append(inference_time)
            self.total_frames += 1
            self.successful_inferences += 1
            
            # Keep only recent timing data
            if len(self.frame_times) > 100:
                self.frame_times.pop(0)
                self.inference_times.pop(0)
                
            # Add timing information
            result.update({
                "frame_time_ms": frame_time * 1000,
                "inference_time_ms": inference_time * 1000,
                "fps": 1.0 / frame_time if frame_time > 0 else 0
            })
            
            return result
            
        except Exception as e:
            self.failed_inferences += 1
            app_logger.error(f"Frame processing failed: {str(e)}")
            
            return {
                "error": str(e),
                "frame_time_ms": (time.time() - frame_start_time) * 1000,
                "successful": False
            }
            
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics."""
        if self.frame_times:
            avg_frame_time = np.mean(self.frame_times)
            avg_inference_time = np.mean(self.inference_times)
            avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        else:
            avg_frame_time = avg_inference_time = avg_fps = 0
            
        return {
            "total_frames": self.total_frames,
            "successful_inferences": self.successful_inferences,
            "failed_inferences": self.failed_inferences,
            "success_rate": self.successful_inferences / max(1, self.total_frames),
            "average_frame_time_ms": avg_frame_time * 1000,
            "average_inference_time_ms": avg_inference_time * 1000,
            "average_fps": avg_fps,
            "preprocessing_steps": self.preprocessing_steps,
            "postprocessing_steps": self.postprocessing_steps
        }


class RealtimeVision:
    """Real-time vision system with neuromorphic processing."""
    
    def __init__(
        self,
        compiled_model: Any,
        camera_interface: str = "opencv",
        preprocessing: str = "normalize",
        buffer_size: int = 10,
        num_worker_threads: int = 2
    ):
        self.compiled_model = compiled_model
        self.camera_interface = camera_interface
        self.buffer_size = buffer_size
        self.num_worker_threads = num_worker_threads
        
        # Vision pipeline
        self.pipeline = VisionPipeline(compiled_model)
        
        # Threading and queues
        self.frame_queue = Queue(maxsize=buffer_size)
        self.result_queue = Queue(maxsize=buffer_size)
        self.worker_threads = []
        self.capture_thread = None
        
        # Control flags
        self._running = False
        self._capture_lock = threading.Lock()
        
        # Monitoring
        self.energy_monitor = EnergyMonitor()
        self.dropped_frames = 0
        self.processing_delays = []
        
        app_logger.info(f"RealtimeVision initialized with {num_worker_threads} workers")
        
    def start_camera(self, camera_id: int = 0, resolution: Tuple[int, int] = None) -> bool:
        """Start camera capture."""
        try:
            if self.camera_interface == "opencv":
                self.camera = cv2.VideoCapture(camera_id)
                
                if resolution:
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
                    
                # Set camera properties for performance
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.camera.set(cv2.CAP_PROP_FPS, 30)
                
                if not self.camera.isOpened():
                    app_logger.error("Failed to open camera")
                    return False
                    
            app_logger.info(f"Camera started: {camera_id}")
            return True
            
        except Exception as e:
            app_logger.error(f"Failed to start camera: {str(e)}")
            return False
            
    def stop_camera(self) -> None:
        """Stop camera capture."""
        if hasattr(self, 'camera'):
            self.camera.release()
            app_logger.info("Camera stopped")
            
    def _capture_frames(self) -> None:
        """Capture frames from camera in separate thread."""
        app_logger.info("Frame capture thread started")
        
        while self._running:
            try:
                ret, frame = self.camera.read()
                
                if not ret:
                    app_logger.warning("Failed to capture frame")
                    continue
                    
                # Add timestamp
                timestamp = time.time()
                
                try:
                    # Non-blocking queue put
                    self.frame_queue.put((timestamp, frame), timeout=0.001)
                except:
                    # Queue full - drop frame
                    self.dropped_frames += 1
                    if self.dropped_frames % 10 == 0:
                        app_logger.warning(f"Dropped {self.dropped_frames} frames due to full buffer")
                        
            except Exception as e:
                app_logger.error(f"Frame capture error: {str(e)}")
                time.sleep(0.1)  # Brief pause before retry
                
        app_logger.info("Frame capture thread stopped")
        
    def _process_frames(self, worker_id: int) -> None:
        """Process frames in worker thread."""
        app_logger.info(f"Worker {worker_id} started")
        
        while self._running:
            try:
                # Get frame from queue
                timestamp, frame = self.frame_queue.get(timeout=1.0)
                
                # Calculate processing delay
                processing_delay = time.time() - timestamp
                self.processing_delays.append(processing_delay)
                
                # Keep only recent delay data
                if len(self.processing_delays) > 100:
                    self.processing_delays.pop(0)
                    
                # Process frame
                result = self.pipeline.process_frame(frame)
                result["timestamp"] = timestamp
                result["worker_id"] = worker_id
                result["processing_delay_ms"] = processing_delay * 1000
                
                # Put result in output queue
                try:
                    self.result_queue.put(result, timeout=0.001)
                except:
                    app_logger.warning("Result queue full, dropping result")
                    
            except Empty:
                continue  # Timeout, check if still running
            except Exception as e:
                app_logger.error(f"Worker {worker_id} error: {str(e)}")
                
        app_logger.info(f"Worker {worker_id} stopped")
        
    def process_stream(
        self,
        fps: float = 30,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        energy_budget: float = 100,  # mW
        duration: Optional[float] = None
    ) -> Dict[str, Any]:
        """Process video stream in real-time."""
        app_logger.info(f"Starting stream processing at {fps} FPS")
        
        self._running = True
        
        # Start energy monitoring
        self.energy_monitor.start_monitoring(energy_budget)
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.capture_thread.start()
        
        # Start worker threads
        for i in range(self.num_worker_threads):
            worker = threading.Thread(target=self._process_frames, args=(i,))
            self.worker_threads.append(worker)
            worker.start()
            
        # Main processing loop
        start_time = time.time()
        frame_interval = 1.0 / fps
        results_processed = 0
        
        try:
            while self._running:
                if duration and (time.time() - start_time) > duration:
                    app_logger.info("Duration limit reached")
                    break
                    
                # Check energy budget
                if self.energy_monitor.is_over_budget():
                    app_logger.warning("Energy budget exceeded")
                    break
                    
                try:
                    # Get result from processing
                    result = self.result_queue.get(timeout=frame_interval)
                    results_processed += 1
                    
                    # Call user callback
                    if callback:
                        try:
                            callback(result)
                        except Exception as e:
                            app_logger.error(f"Callback error: {str(e)}")
                            
                except Empty:
                    continue
                    
        except KeyboardInterrupt:
            app_logger.info("Processing interrupted by user")
            
        finally:
            # Stop processing
            self._running = False
            
            # Wait for threads to finish
            if self.capture_thread:
                self.capture_thread.join(timeout=2.0)
                
            for worker in self.worker_threads:
                worker.join(timeout=2.0)
                
            # Stop energy monitoring
            energy_stats = self.energy_monitor.stop_monitoring()
            
            # Collect final statistics
            processing_stats = self.get_processing_stats()
            
            app_logger.info(f"Stream processing completed: {results_processed} results")
            
            return {
                "results_processed": results_processed,
                "duration": time.time() - start_time,
                "processing_stats": processing_stats,
                "energy_stats": energy_stats,
                "dropped_frames": self.dropped_frames
            }
            
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get real-time processing statistics."""
        pipeline_stats = self.pipeline.get_performance_stats()
        
        additional_stats = {
            "dropped_frames": self.dropped_frames,
            "average_processing_delay_ms": np.mean(self.processing_delays) * 1000 if self.processing_delays else 0,
            "frame_queue_size": self.frame_queue.qsize(),
            "result_queue_size": self.result_queue.qsize(),
            "worker_threads": self.num_worker_threads,
            "camera_interface": self.camera_interface
        }
        
        return {**pipeline_stats, **additional_stats}
        
    def cleanup(self) -> None:
        """Cleanup vision system resources."""
        self._running = False
        self.stop_camera()
        
        # Clear queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
                
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except Empty:
                break
                
        app_logger.info("RealtimeVision cleanup completed")


class EnergyMonitor:
    """Monitor energy consumption during processing."""
    
    def __init__(self):
        self.monitoring = False
        self.energy_budget = 0
        self.energy_consumed = 0
        self.power_samples = []
        self.start_time = 0
        
    def start_monitoring(self, energy_budget: float) -> None:
        """Start energy monitoring."""
        self.energy_budget = energy_budget
        self.energy_consumed = 0
        self.power_samples = []
        self.start_time = time.time()
        self.monitoring = True
        
        app_logger.info(f"Energy monitoring started with budget: {energy_budget} mW")
        
    def update_power_consumption(self, power_mw: float) -> None:
        """Update power consumption measurement."""
        if self.monitoring:
            self.power_samples.append((time.time(), power_mw))
            
            # Calculate energy consumption (integrate power over time)
            if len(self.power_samples) > 1:
                prev_time, prev_power = self.power_samples[-2]
                curr_time, curr_power = self.power_samples[-1]
                
                dt = curr_time - prev_time
                avg_power = (prev_power + curr_power) / 2
                energy_delta = avg_power * dt  # mJ
                
                self.energy_consumed += energy_delta
                
    def is_over_budget(self) -> bool:
        """Check if energy consumption is over budget."""
        if not self.monitoring:
            return False
            
        # Estimate current energy consumption
        elapsed_time = time.time() - self.start_time
        estimated_energy = self.energy_budget * elapsed_time / 1000  # Convert to mJ
        
        return self.energy_consumed > estimated_energy
        
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop energy monitoring and return statistics."""
        self.monitoring = False
        
        total_time = time.time() - self.start_time
        avg_power = np.mean([p for _, p in self.power_samples]) if self.power_samples else 0
        
        stats = {
            "total_energy_consumed_mj": self.energy_consumed,
            "energy_budget_mj": self.energy_budget * total_time / 1000,
            "average_power_mw": avg_power,
            "monitoring_duration_s": total_time,
            "energy_efficiency": self.energy_consumed / max(total_time, 1),
            "budget_utilization": self.energy_consumed / max(self.energy_budget * total_time / 1000, 1)
        }
        
        app_logger.info(f"Energy monitoring stopped: {stats['total_energy_consumed_mj']:.3f} mJ consumed")
        
        return stats