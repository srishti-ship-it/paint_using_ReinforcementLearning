# Reinforcement Learning for Autonomous Painting Using Multi-Objective Reward Functions

This repository contains the code and report for my course project on **stroke-based autonomous painting using Deep Reinforcement Learning**.

An RL agent learns to paint images step-by-step using strokes, instead of generating the whole image in one pass. The project supports **three different painting modes**, each with its own preprocessing and reward function:

- 🎨 **Paint-by-Number** – the agent fills color-quantized regions of the target image.
- 🖌️ **Stylization** – the agent produces a smooth, painterly version of a photo using low-frequency similarity.
- ✏️ **Contour Reconstruction** – the agent learns to draw object outlines by matching edge maps.

The agent uses a **continuous Actor–Critic network** and learns by interacting with a custom painting environment implemented in Python and PyTorch.:contentReference[oaicite:1]{index=1}

---

## Project Structure

- `paint_rl_colab.ipynb`  
  Google Colab notebook with:
  - Environment (`PaintEnvMulti`)
  - Actor–Critic model
  - Mode selection: `paint_by_number`, `stylization`, `contour`
  - Training loop and GIF generation

- `Reinforcement Learning for Autonomous Painting Using Multi Objective Reward Functions.pdf`  
  Full project report (abstract, methodology, results, discussion, flowchart).:contentReference[oaicite:2]{index=2}

- `Flowchart_RL_Painting.png`  
  High-level flowchart of the RL painting pipeline.

- `gif_outputs/` (optional)  
  Example training GIFs showing how the canvas evolves stroke-by-stroke for each mode.

---

## Methods (Short Summary)

1. **Environment**
   - Canvas: 32×32 RGB image, initialized white.
   - Target: uploaded image or demo shape.
   - Observations:
     - 6 channels: `canvas RGB + target RGB` (for paint-by-number, stylization, contour).
   - Actions: 8D continuous vector  
     `[x0, y0, x1, y1, r, g, b, thickness]`.

2. **Mode-specific preprocessing**
   - **Paint-by-Number**: K-Means color quantization → region labels + palette.
   - **Stylization**: Gaussian blur of target → low-frequency target.
   - **Contour**: grayscale + Sobel edge map.

3. **Reward functions**
   - Paint-by-Number: negative MSE between region color and palette color.
   - Stylization: combination of blurred MSE and color-mean similarity.
   - Contour: negative MSE between canvas edge map and target edge map.:contentReference[oaicite:3]{index=3}

4. **RL Algorithm**
   - Actor–Critic (A2C-style) with:
     - Gaussian policy over stroke parameters.
     - Value head predicting expected return.
   - Loss = policy loss + value loss – entropy bonus.

---

## How to Run (Colab)

1. Open `paint_rl_colab.ipynb` in **Google Colab**.
2. Run the **install** cell to install dependencies.
3. Upload a target image (e.g., `penguin.png`) using the upload cell.
4. Set the mode:
   ```python
   MODE = 'paint_by_number'  # or 'stylization' or 'contour'
