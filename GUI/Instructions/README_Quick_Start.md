# FACT GUI quick start

Follow this illustrated workflow with the MouseBrainCortex demo, then choose **Instruction** (menu) -> **New Data Trial** to run an instruction for new data, and follow the same sequence for your own recording.

### 1. Launch and load the demo

For a standalone first trial, download [FACT-Pipeline_light.zip](https://drive.google.com/file/d/1moZK8P6hIuX0UCIKTAgESQTkQ63N6kIB/view?usp=drive_link) and extract the entire ZIP. If using a cloned repository instead, run `GUI\Download FACT-GUI_(RUN ME).bat` and choose **GUI plus basic demo data**. Open `GUI\FACT-Pipeline.exe`, keeping `FACT-Pipeline.runtime` beside it. Choose **Demo → MouseBrainCortex**. The TIFF and saved parameters load together; keep those settings for your first run.

![1. Load MouseBrainCortex from the Demo menu and confirm the input](images/01-load-demo.png)

To install all demos manually, download `GUI.zip` and `data.zip` individually
from the [ZIP download folder](https://drive.google.com/drive/folders/1c3iSZWt9nLQAtM5WBqrPggZ9tPRCRXKl?usp=sharing), then extract their contained
`GUI/` and `data/` folders into the same parent folder. Avoid `GUI/GUI/` and
`data/data/` nesting. In the light package, you can instead run
`data\Download Additional Demo Data_(RUN ME).bat` to choose CA1+CA3 together,
Meso, Rhesus, multiple groups, or all remaining demos. Existing files are
skipped. Restart the GUI after downloading and select a dataset through **Demo**.

### 2. Run FACT Network

Click **RUN FACT NETWORK** and follow **STATUS** until inference completes. Move **Frame** to compare Raw video and FACT Network Inference at the same time point. Scroll to zoom, drag to pan, and double-click to fit; both views move together.

![2. Run the network and compare the synchronized video views](images/02-network.png)

### 3. Run post-processing

Scroll down the left sidebar to **RUN POST-PROCESSING**. Click it and wait for Page 2 to open automatically. The demo already supplies post-processing values; **Auto** is unavailable until its first post-processing run finishes.

![3. Find the post-processing controls and run button in the sidebar](images/03-postprocessing.png)

### 4. Inspect masks and traces

Click a colored neuronal mask to display its ROI, background, and background-removed traces. Adjust **Opacity** or hide the colored overlay to inspect the underlying standard-deviation image. Scroll, drag, and double-click also zoom, pan, and reset the trace plot.

![4. Select a neuron and inspect its masks and traces](images/04-masks-traces.png)

### 5. Export results

Open **Export**, choose NumPy, MATLAB, or HDF5, select the required components, then click **Choose directory...**. Use **File → Export Results...** when you also need network products and the effective run configuration.

![5. Choose an export format, components, and destination](images/05-export.png)

### 6. Use your own recording

Choose **Load File** for a grayscale `(T, H, W)` TIFF, or **Load Folder** to concatenate compatible TIFFs along time. Set neuronal radii in pixels and the acquisition frame rate. Choose Fast or Dense and an appropriate **Normalization mode**; use **Automatic estimate** for ordinary input. Click **Auto**, then repeat network inference, post-processing, inspection, and export.

![6. Set input, neuronal size, normalization, frame rate, and automatic post-processing values](images/06-own-data.png)
