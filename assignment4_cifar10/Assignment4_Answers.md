# Assignment 4 Theory Answers

## Question 1

For embedding dimension `d`, timestep `t`, and pair index
`i = 0, 1, ..., d/2 - 1`, the standard sinusoidal timestep embedding is:

\[
\mathrm{emb}_{2i}(t)=\sin\left(\frac{t}{10000^{2i/d}}\right)
\]

\[
\mathrm{emb}_{2i+1}(t)=\cos\left(\frac{t}{10000^{2i/d}}\right)
\]

Each pair uses a different frequency. Small values of `i` change quickly with
`t`, while larger values change more slowly.

## Question 2

Here `d = 8`, `t = 1`, and the maximum period is 10000. There are four
sine/cosine pairs. Their denominators are:

\[
10000^{0/8}=1,\quad
10000^{2/8}=10,\quad
10000^{4/8}=100,\quad
10000^{6/8}=1000.
\]

Therefore, using the standard alternating sine/cosine convention:

\[
[\sin(1),\cos(1),\sin(0.1),\cos(0.1),
\sin(0.01),\cos(0.01),\sin(0.001),\cos(0.001)]
\]

Approximately:

\[
[0.841471,\ 0.540302,\ 0.099833,\ 0.995004,\
0.010000,\ 0.999950,\ 0.001000,\ 1.000000].
\]

## Question 3

Both Transformer positional encoding and diffusion timestep embedding use
sinusoids with several frequencies to turn a scalar index into a vector. This
lets the neural network distinguish different positions or noise levels while
retaining a smooth relationship between nearby values.

The key difference is what the scalar represents and how the embedding is used:

- In a Transformer, it represents a token's position in a sequence and is
  normally added to token embeddings.
- In a diffusion model, it represents the current noise timestep. It conditions
  the denoising UNet, commonly through projected additions inside residual
  blocks or by concatenating an embedding with feature maps.

## Question 4

Each stride-2 downsampling block halves the height and width:

\[
64\times64 \rightarrow 32\times32 \rightarrow 16\times16
\rightarrow 8\times8.
\]

Therefore, the bottleneck spatial resolution is **8 x 8**.

## Question 5

The UNet receives the noisy image \(x_t\) and timestep \(t\). It outputs an
estimate of the noise that was added:

\[
\epsilon_\theta(x_t,t).
\]

During training, the true Gaussian noise \(\epsilon\) is known because it was
sampled when constructing \(x_t\). The prediction is compared with the true
noise, commonly using mean squared error:

\[
L = \mathbb{E}_{x_0,t,\epsilon}
\left[\lVert \epsilon-\epsilon_\theta(x_t,t)\rVert^2\right].
\]

The supplied practical follows the same noise-prediction idea; its selected
loss function may be L1 instead of MSE. The objective is still to make the
predicted noise match the actual added noise.

## Question 6: Basic Gradient Calculations

Given:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x**2 + 3 * x
y.backward()
```

### a)

\[
y=x^2+3x
\]

\[
\frac{dy}{dx}=2x+3.
\]

At \(x=2\):

\[
\frac{dy}{dx}=2(2)+3=7.
\]

The result is:

```text
x.grad = tensor([7.])
```

### b)

With `requires_grad=False`, PyTorch does not build a gradient graph for `x`.
Consequently, `y` also does not require gradients, and calling `y.backward()`
raises a runtime error because there is no graph to differentiate.

### c)

No. For `torch.tensor`, the default is:

```python
requires_grad=False
```

Therefore, gradients are not tracked unless the flag is explicitly set to
`True`.

## Question 7: Introduce Weights

Given:

```python
x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([1.0, 3.0])
y = w[0] * x**2 + w[1] * x
y.backward()
```

### a)

The result of:

```python
print("w.grad =", w.grad)
```

is:

```text
w.grad = None
```

This happens because `w` was created with the default
`requires_grad=False`. PyTorch therefore does not store gradients for `w`.

### b)

Set `requires_grad=True` when creating `w`:

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([1.0, 3.0], requires_grad=True)

y = w[0] * x**2 + w[1] * x
y.backward()

print("x.grad =", x.grad)
print("w.grad =", w.grad)
```

The derivatives with respect to the weights are:

\[
\frac{\partial y}{\partial w_0}=x^2=4
\]

and

\[
\frac{\partial y}{\partial w_1}=x=2.
\]

Therefore:

```text
w.grad = tensor([4., 2.])
```

### c)

No. A tensor created without the flag uses `requires_grad=False`, so its
gradients are not tracked.

## Question 8: Breaking the Graph

`detach()` returns a tensor that shares the same value but is disconnected from
the previous computational graph. Therefore, `z` does not remember that it came
from `y` or `x`. Since `w` is computed only from detached `z`, `w` does not
require gradients, so `w.backward()` fails.

The simplest correction is to keep the variable `z` but not detach it:

```python
import torch

x = torch.tensor([1.0], requires_grad=True)
y = x * 3
z = y
w = z * 2
w.backward()

print(x.grad)
```

Because \(w=6x\), the result is:

```text
tensor([6.])
```

If a detached forward value must explicitly appear in the calculation, a
straight-through form can be used:

```python
z_detached = y.detach()
z = z_detached + (y - y.detach())
w = z * 2
w.backward()
```

The forward value is unchanged, but the `(y - y.detach())` term supplies a
path for the gradient back to `x`.

## Question 9: Gradient Accumulation

After the first backward pass:

\[
y_1=2x \quad\Rightarrow\quad \frac{dy_1}{dx}=2,
\]

so:

```text
After first backward: x.grad = tensor([2.])
```

PyTorch accumulates gradients instead of replacing them. The second backward
pass adds the new gradient:

\[
y_2=3x \quad\Rightarrow\quad \frac{dy_2}{dx}=3.
\]

Thus:

```text
After second backward: x.grad = tensor([5.])
```

To avoid accumulation, clear the gradient before the next backward pass:

```python
x.grad.zero_()
```

or:

```python
x.grad = None
```

For neural-network parameters, the normal pattern is:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```
