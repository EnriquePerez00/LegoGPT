# LegoGPT Agent Rules

## Physical Connection Constraints
- **Stud-to-Socket Rule**: Lego parts can only connect if there is a matching physical alignment between a stud (conos de encaje) and a socket (hueco). Connections without matching studs and sockets are strictly forbidden.
- **Surface and Height Level Rules**: 
  - All parts must reside at or above the ground surface level in the normalized coordinate system (Y <= 0.0 in standard LDraw coordinates, where min(Y_bottom) is normalized to ground level Y = 0.0).
  - During intermediate build steps or when the assembly is rotated/flipped (e.g. inverted steps), parts may temporarily be placed under existing layers; upon step completion, the whole assembly must be renormalized so that min(Y) aligns with the construction ground (Y = 0.0).
  - At most 4 parts are allowed to touch the ground/surface directly (|Y_world| <= 1.0 after normalization).
- **Physical Overlaps**: Overlaps between oriented bounding boxes are strictly prohibited in both physical database generation and 3D web visualizations.

## Hardware Acceleration & Computing Optimization Rules (Apple Silicon M4 / 48 GB RAM)
Whenever creating, modifying, refactoring, or running any computation, parsing, ML/DL inference, rendering, visual scoring, search, or batch data processing in this repository, you **MUST** leverage the local hardware profile:
- **Architecture & System**: macOS, Apple Silicon M4 Chip (12 physical/performance-efficient cores, ARM64), 48 GB Unified Memory, Metal-enabled GPU.

### 1. Multi-Core Parallelism & Concurrency (Safety Buffer Reserved)
- **Dynamic Worker Allocation**: Never hardcode thread or worker counts. Compute workers dynamically:
  ```python
  max_workers = max(1, (os.cpu_count() or 12) - 2)  # Reserves 2-4 cores for OS responsiveness
  ```
- **CPU-bound vs I/O-bound Workloads**:
  - **CPU-bound operations** (PDF geometry/layout parsing, OBB collision checks, spatial hypothesis generation, 3D mesh slicing, graph validation): Use `concurrent.futures.ProcessPoolExecutor` or multiprocessing with forkserver/spawn safe paradigms on macOS.
  - **I/O, Network & DB extraction** (web scraping, image downloading, SQLite reads, PDF raster rendering when memory allows): Use `concurrent.futures.ThreadPoolExecutor` with worker pools scaled between `(os.cpu_count() - 2)` and `20`.
- **Batch Processing**: Always chunk large iterable collections (e.g., thousands of steps, dataset elements, OMR files) into vectorized or worker-batched tasks rather than sequential single-item processing.

### 2. Apple Silicon GPU & Neural Engine Acceleration (MPS / Metal)
- **PyTorch & Torchvision Models**: Verify and prioritize `torch.device("mps")` over CPU for all deep learning models, embeddings, classifiers, and vision pipelines (e.g., ResNet, YOLO, MobileNet, custom CNNs):
  ```python
  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  ```
- **Batching on MPS**: Avoid processing single images one by one through the GPU. Collate images into mini-batches (`batch_size=16`, `32`, or `64`) using PyTorch tensors and send the entire batch to MPS simultaneously.
- **Metal Performance Shaders (MPS) Graph Compilation**: Use PyTorch vectorization (`torch.matmul`, `torch.cdist`, `torch.cosine_similarity`, `torch.nn.functional`) instead of manual Python `for` loops when computing pairwise distances, feature similarities, or collision matrices across large candidate spaces.

### 3. Memory Management & Dynamic Scaling (48 GB Unified Memory)
- **High-Throughput In-Memory Caching**: Take advantage of the 48 GB RAM to cache heavy geometry meshes, connector databases, and visual template embeddings in memory (e.g., in-memory dictionaries, LRU caches, or shared memory pools) rather than reloading or recomputing from disk repeatedly.
- **Dynamic Memory Safety Guard**:
  - Monitor memory headroom with `psutil.virtual_memory()` when orchestrating massive parallel extractions or dataset builds.
  - Keep a memory buffer (limit application memory consumption to 75–80% of available RAM, ~36–38 GB) to ensure macOS UI and background tasks remain responsive without memory swapping.
- **Streamlined Generator Pipelines**: For massive PDF manual renderings or video extractions, yield processed frames or page data iteratively or clean up unneeded PyMuPDF `fitz.Pixmap` / OpenCV buffers explicitly using `del` or context managers to prevent uncollectible native memory spikes.
