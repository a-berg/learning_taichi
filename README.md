# Taichi explorations

In this repo you can find a couple examples in [Taichi](https://www.taichi-lang.org/)
regarding the computation of [Julia sets](https://docs.taichi-lang.org/docs/hello_world)
and solving the [Gray-Scott
equations](https://en.wikipedia.org/wiki/Reaction%E2%80%93diffusion_system) in 2d.

<!-- You can find a blog post here: post. -->

## What is Taichi?

Taichi is a compiler and a library. The library adds python bindings to C++ to allow the
programmer for transparent use of advanced parallelization and GPGPU; whereas the compiler
let's you compile said python code with llvm and reap the benefits of:

- coding in a high-level, high-productivity language like python
- running in a optimized framework for parallel CPU, GPU computing.

In other words, Taichi sort of extends the Python language and overloads it with
high-performance parallel capabilities.

The main change Taichi will make in your code is: you get `for` loops back!

### Aside: why not Numba?

Numba is very similar in purpose and implementation to Taichi. Both are "extension
languages" to base Python using decorators to allow for easy parallelization, and both
target GPU and CPU parallelization.

I'm not familiar with Numba, to be honest. However, some benchmarks seem to indicate
that Taichi is more efficient, in fact, Taichi code usually outperforms native CUDA
(which is impressive); plus, Taichi lets you target CPU and GPU with a single kernel
(instead of having to write different functions as in Numba). Finally, Taichi has
autodiff out of the box.

In my opinion, Taichi is a more modern, and seemingly better, option than Numba for the
use case of high-performance computing.

## What can you find in this repo?

I was interested in Taichi because I do ML, and it offered potential for integration
with Pytorch, with interesting use cases for computer vision, LiDAR and hybridizing
mathematical optimization with DL.

As a first contact with the language, I decided to follow the ["hello
world"](https://docs.taichi-lang.org/docs/hello_world) example to compute Julia sets and
then attempt to (by myself) code a simple solver for the Gray Scott system of equations,
adapting the code from a previous numpy code I had.

> disclaimer: you can find another G-S implementation in the [Taichi
> blog](https://docs.taichi-lang.org/blog/accelerate-python-code-100x#reaction-diffusion-equations),
> which I didn't know of at the time of writing my first Taichi implementation. After
> discovering it, I used it to do the color-rendering part and debugging a performance
> issue.

### Julia sets computation

It's basically a 1:1 writing of the mentioned "hello world" example. A result can be seen here:

![Julia sets animation](./results/julia/video.gif)

To run the code, use:

```
ti run src/julia_ti.py
```

### Solving Gray-Scott model for reaction-diffusion systems

I adapted the code from a previous one I had which used numpy, then explored the
Object-Oriented capabilities of Taichi to refactor the code into something (marginally)
better.

You can see the results here:

![Reaction diffusion system evolution](./results/reaction_diffusion/video.gif)

and run the code using:

```
ti run src/gray_scott_ti.py
```

The original numpy code can be inspected, too: [original](./src/gray_scott.py).

## Conclusions

This is merely a simple initial contact with Taichi. I have liked the library a lot and
would like to use it more, so I probably will.

If you have any comment, open a discussion in the repository!
