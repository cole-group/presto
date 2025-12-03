# Theory

The accuracy of transferable molecular mechanics force fields is often limited by their lack of transferability, rather than their functional form. ``bespokefit_smee`` aims to generate accurate molecular mechanics force field parameters specifically for your molecule/ molecules of interest. This is done by fitting parameters to energies and forces from a machine learning potential, which loses little accuracy compared to the QM method it was trained on, but is orders of magnitude faster:

![Alt text](images/workflow-summary.png)

## Initial force field

The fit can be started from any standard OpenFF, force field. Only the valence parameters (bonds, angles, proper torsions, and improper torsions) are trained, while the Lennard-Jones terms and changes are left unaltered. The functional form of the valence terms are:

- Bonds and angles are defined by a harmonic function,
$u(x;k,x_0)=\frac{k}{2}\left(x-x_0\right)^2$,
where the position of the minimum, $x_0$, and the magnitude, $k$, are the fitting parameters.
- Proper and improper torsions are defined by a set of cosine functions,
$u_p(\phi;k,\phi_0)=k\left(1+\cos{\left(p\phi-\phi_0\right)}\right)$,
where the phase, $\phi_0$, and the magnitude, $k$, are the fitted parameters. Here, proper torsions are expanded to include four periodicities, whereas improper torsions include only one. It is also noted that for symmetry, the phase $\phi_0$ is expected to be either 0 or $\pi$

## Bespoke parameter generation

The parameters in an OpenFF SMIRNOFF force field are assigned to specific bonds, angles, etc. using "SMIRKS" (really tagged SMARTS) patterns which are generally very non-specific. By default, we generate extremely specific "SMIRKS" patterns which specify the entire molecule of interest. 

## Sampling

The molecule is sampled using high-temperature molecular dynamics. By default, this is performed at 500 K, using the input molecular mechanics force field, well-tempered metadynamics is applied to all rotatable bonds to enhance sampling of diverse conformers and torsional barriers. The sampling is started from several different conformers generated with ``RDKit``'s ``ETKDG`` algorithm. 

## Energy and force evaluation

Snapshots are saved from the molecular dynamics and the energies and forces of each are computed using a machine-learning potential such as Egret-1 or AIMNet2. By default, energies are offset by their mean before training.

## Training

The molecular mechanics force field parameters are optimised to reproduce the energies and forces from the machine learning potential. A regularisation penalty is also applied for deviations of the improper and proper torsion parameters from their starting point, as we found this was important to avoid a number of torsion-barrier outliers. By default, the Adam optimiser is used. A technicality of training is that we linearise the harmonic potentials (bonds and angles) to stabilise fitting -- see the footnote.

## Iterations

Optionally, the user can perform iterative fitting, where the molecular mechanics force field (which is used for sampling) is iteratively refined and sampled.

## Final force field

The bespoke parameters are added on to the end of the input force field and this is saved (see ``bespoke_ff.offxml`` in the relevant output directory). Because parameters lower down the ``.offxml`` file are given higher priority, these parameters are used instead of the original non-specific parameters from the input force field when you parameterise your molecule of interest.

---

### Footnote

To stabilise and speed up convergence of the parameter fitting, harmonic potentials are *linearized*.

The linearization of the harmonic terms followed the approach by [espaloma](https://doi.org/10.1039/D2SC02739A), where the minimum is assumed to be within a window given by $x_1$ and $x_2$, such that the fitting parameters may by remapped onto linear terms,

$$k_1=k\frac{x_2-x_0}{x_2-x_1} \quad\text{and}\quad k_2=k\frac{x_0-x_1}{x_2-x_1}$$

These terms give the original parameters via,

$$k=k_1+k_2 \quad\text{and}\quad x_0=\frac{k_1x_1+k_2x_2}{k_1+k_2}$$

Crucially, the gradient along $k_1$ and $k_2$ behaves more reliably and so the parameters minimize faster.
