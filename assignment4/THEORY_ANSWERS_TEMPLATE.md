# Assignment 4 Theory Answers

The screenshots only show the first page of the PDF, so the exact theory
questions are not visible here. Paste the full theory questions into this file
and answer them before submission.

## Useful building blocks to include if relevant

### Forward diffusion

A clean image x_0 is gradually corrupted by Gaussian noise:

q(x_t | x_0) = N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)

This can be implemented as:

x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) epsilon,
where epsilon ~ N(0, I).

### Noise-prediction training objective

The model epsilon_theta(x_t, t) is trained to predict the noise that was added:

L = E[ || epsilon - epsilon_theta(x_t, t) ||_2^2 ].

### EBM sampling

An EBM defines an energy E_theta(x). Low-energy images are more likely.
Sampling can be done with Langevin dynamics by updating the image itself:

x <- x - eta * grad_x E_theta(x) + sigma * N(0, I).

This is why the input image tensor requires gradients during sampling.
