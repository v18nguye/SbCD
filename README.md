# Symmetry-Breaking De Novo Crystal Generation via Markovian Jump Diffusion
#### Authors: [Van Khoa Nguyen](https://scholar.google.com/citations?user=UDVkJKcAAAAJ&hl=en&authuser=1), [Alexandros Kalousis](https://scholar.google.com/citations?hl=en&user=uVkn9UEAAAAJ&view_op=list_works&sortby=pubdate)

## Abstract
Generating crystals has recently attracted significant interest due to their broad applications in materials science. However, existing generative models struggle to produce complete crystallographic specifications, limiting their ability to capture global symmetry and structural dependencies. In particular, current state-of-the-art approaches generate crystals only up to site symmetries and rely on sampling space groups from empirical distributions during generation. Inspired by *spontaneous symmetry breaking* in physics, where crystals break symmetries under external conditions, we propose a novel diffusion-based framework that generates full structure specifications by reversing from the lowest-symmetry priors. Our method leverages a Markovian jump-diffusion process to model these symmetry-breaking dynamics, enabling it to traverse different space groups in a physically motivated manner. Our model, dubbed *Symmetry-breaking Crystal Diffusion* (SbCD), introduces a principled approach to explicitly incorporate inter-space-group transitions into the generative process. In de novo crystal generation experiments on MP20, SbCD outperforms its symmetry-preserving generative counterpart by a substantial margin, offering a promising perspective for generative modeling of crystalline materials.

<p align="center">
  <img src="assets/main.png" alt="Overview of the proposed method" width="900">
</p>

🔔 **Code release is planned upon paper acceptance. Stay tuned!**
